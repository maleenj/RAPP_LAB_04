"""VAM TM12S inference node — visualization variant.

A thin SUBCLASS of the working VAMTM12SInferenceNode that ADDS neural-network
"brain scan" publishing for the workshop, without touching the original node:

  * Forward-hooks capture internal activations during the normal predict() call:
      - decoder_xattn      [n_dec, heads, T_out, T_in]  (cross-attention: action→perception)
      - encoder_selfattn   [n_enc, heads, T_in,  T_in]  (temporal self-attention)
      - encoder_out        [T_in, d_model]              (latent "situation" state)
      - decoder_out        [T_out, d_model]             (motor-plan latent)
  * Optional saliency (publish_saliency:=true) runs a separate gradient-enabled
    forward to attribute the action back onto the human's keypoints over time:
      - input_saliency     [T_in, n_keypoints]
  * All of the above are packed into ONE std_msgs/Float32MultiArray on
    /vam/activations, with layout.dim labels "name:d0xd1x..." so the viz bridge's
    multiarray_named serializer reconstructs the named sub-tensors for clients.

The control path is UNCHANGED: we never override _timer_cb. Hooks fill a cache
during the parent's predict(); an independent timer drains and publishes it.

Why a separate node/executable (not an edit to vam_tm12s_node.py)? So the proven
pipeline is never modified — exactly like the existing vam_tm12s_node_diag.py fork.

Note: forcing attention weights switches MultiheadAttention to its unfused math
path, which shifts predictions by ~1e-6 rad (≈6e-5 deg) — numerically negligible,
verified. Saliency adds a small extra forward; it is OFF by default.
"""

import rclpy
import torch
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from vam_inference.vam_tm12s_node import VAMTM12SInferenceNode


# Names this node knows how to capture. activation_set selects from these.
_ATTENTION_SETS = ("encoder_selfattn", "decoder_selfattn", "decoder_xattn")
_LATENT_SETS = ("encoder_out", "decoder_out")


class VAMTM12SVizNode(VAMTM12SInferenceNode):
    def __init__(self):
        # Build the full, unmodified inference node first.
        super().__init__()

        # --- Visualization parameters (additive; defaults safe) ---
        self.declare_parameter("publish_activations", True)
        self.declare_parameter(
            "activation_set",
            ["decoder_xattn", "encoder_selfattn", "encoder_out", "decoder_out"],
        )
        self.declare_parameter("publish_saliency", False)
        self.declare_parameter("activation_publish_rate_hz", 15.0)

        self._publish_acts = self.get_parameter("publish_activations").value
        self._activation_set = list(self.get_parameter("activation_set").value)
        self._publish_saliency = self.get_parameter("publish_saliency").value
        pub_rate = float(self.get_parameter("activation_publish_rate_hz").value)

        self._hook_handles = []
        self._acc = {}            # per-forward accumulator (written by hooks)
        self._act_latest = None   # finalized snapshot: list of (name, shape, flat)
        self._act_dirty = False

        if not self._publish_acts:
            self.get_logger().info("Viz node: publish_activations=false (acts as the base node).")
            return

        self._act_pub = self.create_publisher(Float32MultiArray, "/vam/activations", 10)
        self._viz_ready = True
        self._register_activation_hooks()

        period = 1.0 / max(pub_rate, 1.0)
        self._act_timer = self.create_timer(period, self._publish_activations_cb)
        self.get_logger().info(
            f"Viz node: publishing /vam/activations set={self._activation_set} "
            f"saliency={self._publish_saliency} @ {pub_rate:.0f}Hz"
        )

    # --- Re-attach hooks after a runtime model switch -------------------- #
    def _load_pipeline(self, config) -> None:
        super()._load_pipeline(config)
        # Guard: during super().__init__() this runs before viz setup exists.
        if getattr(self, "_viz_ready", False) and self._publish_acts:
            self._register_activation_hooks()

    # --- Hook registration ---------------------------------------------- #
    def _register_activation_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        model = self._model.model  # the underlying nn.Module
        want = self._activation_set

        # Clear the accumulator at the start of every forward pass.
        self._hook_handles.append(
            model.register_forward_pre_hook(self._forward_pre_clear)
        )

        if "encoder_selfattn" in want:
            for layer in model.encoder.layers:
                self._add_attn_hooks(layer.self_attn, "encoder_selfattn")
        if "decoder_selfattn" in want:
            for layer in model.decoder.layers:
                self._add_attn_hooks(layer.self_attn, "decoder_selfattn")
        if "decoder_xattn" in want:
            for layer in model.decoder.layers:
                self._add_attn_hooks(layer.multihead_attn, "decoder_xattn")
        if "encoder_out" in want:
            self._hook_handles.append(
                model.encoder.register_forward_hook(self._make_latent_hook("encoder_out"))
            )
        if "decoder_out" in want:
            self._hook_handles.append(
                model.decoder.register_forward_hook(self._make_latent_hook("decoder_out"))
            )

        # Finalize the snapshot after the whole model forward completes.
        self._hook_handles.append(model.register_forward_hook(self._forward_finalize))

    def _add_attn_hooks(self, attn_module, name: str) -> None:
        self._hook_handles.append(
            attn_module.register_forward_pre_hook(_force_need_weights, with_kwargs=True)
        )
        self._hook_handles.append(
            attn_module.register_forward_hook(self._make_attn_hook(name))
        )

    # --- Hook callbacks -------------------------------------------------- #
    def _forward_pre_clear(self, module, args):
        self._acc = {}

    def _make_attn_hook(self, name: str):
        def hook(module, inp, out):
            # out = (attn_output, attn_weights[B, heads, L, S])
            w = out[1]
            if w is None:
                return
            self._acc.setdefault(name, []).append(w.detach().to("cpu", torch.float32))
        return hook

    def _make_latent_hook(self, name: str):
        def hook(module, inp, out):
            self._acc[name] = out.detach().to("cpu", torch.float32)
        return hook

    def _forward_finalize(self, module, inp, out):
        """Assemble the per-forward accumulator into a published snapshot."""
        snapshot = []
        for name in self._activation_set:
            if name in _ATTENTION_SETS:
                layers = self._acc.get(name)
                if not layers:
                    continue
                t = torch.stack([w[0] for w in layers], dim=0)  # [n_layers, heads, L, S]
            elif name in _LATENT_SETS:
                t = self._acc.get(name)
                if t is None:
                    continue
                t = t[0]  # drop batch -> [T, d_model]
            else:
                continue
            snapshot.append((name, tuple(t.shape), t.flatten().tolist()))
        self._act_latest = snapshot
        self._act_dirty = True

    # --- Saliency (optional, separate grad forward) --------------------- #
    def _compute_saliency(self):
        if not self._assembler.is_ready():
            return None
        try:
            model = self._model.model
            cfg = self._model.model_config
            inp = self._assembler.get_input_tensor().to(self._model.device).clone()
            inp.requires_grad_(True)
            with torch.enable_grad():
                out = model(inp)
                out.norm().backward()
            grad = inp.grad.detach().abs()[0].to("cpu", torch.float32)  # [T_in, input_dim]
            skel_dim = cfg.skeleton_dim
            # per-keypoint magnitude: sum the x/y/z of each keypoint
            kpts = grad[:, :skel_dim].reshape(grad.shape[0], -1, 3).sum(dim=-1)  # [T_in, K]
            return ("input_saliency", tuple(kpts.shape), kpts.flatten().tolist())
        except Exception as exc:  # never let viz break the node
            self.get_logger().warn(f"saliency failed: {exc}")
            return None

    # --- Publish timer --------------------------------------------------- #
    def _publish_activations_cb(self) -> None:
        if not self._act_dirty or self._act_latest is None:
            return
        entries = list(self._act_latest)
        self._act_dirty = False

        if self._publish_saliency:
            sal = self._compute_saliency()
            if sal is not None:
                entries.append(sal)

        if not entries:
            return

        msg = Float32MultiArray()
        layout = MultiArrayLayout()
        data = []
        for name, shape, flat in entries:
            dim = MultiArrayDimension()
            dim.label = f"{name}:{'x'.join(str(s) for s in shape)}"
            dim.size = len(flat)
            dim.stride = len(flat)
            layout.dim.append(dim)
            data.extend(flat)
        msg.layout = layout
        msg.data = data
        self._act_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VAMTM12SVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _force_need_weights(module, args, kwargs):
    """forward_pre_hook: make MultiheadAttention return per-head weights."""
    kwargs = dict(kwargs)
    kwargs["need_weights"] = True
    kwargs["average_attn_weights"] = False
    return args, kwargs


if __name__ == "__main__":
    main()
