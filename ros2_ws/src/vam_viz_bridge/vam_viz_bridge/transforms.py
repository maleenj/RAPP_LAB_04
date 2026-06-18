"""Derived-metric transforms for the viz bridge.

This module is where the *explainability math* lives — deliberately in the
modular viz bridge (rapp_viz docker), NOT in the inference node. The inference
node publishes only RAW activation tensors; the bridge computes derived metrics
from them and republishes each as its own channel. You can add/tweak metrics and
restart only `rapp_viz` — the robot pipeline is never touched.

Each transform takes the deserialized `tensors` dict produced by
serializers.serialize_multiarray_named (i.e. {name: {"shape": [...], "data": [...]}})
and returns a JSON-friendly dict fragment (the node adds channel + stamp).

Add a new metric = add one function + register it in TRANSFORMS. Reference it
from the `derived:` section of bridge_channels.yaml.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _as_array(tensor: dict) -> np.ndarray:
    """Reconstruct an ndarray from a {"shape","data"} sub-tensor."""
    return np.asarray(tensor["data"], dtype=np.float32).reshape(tensor["shape"])


def attention_entropy(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Per-layer mean attention entropy for any attention tensors present.

    Entropy of each attention distribution (over the keys axis) measures how
    *focused* vs *diffuse* the model's attention is: low = laser-focused on a
    few timesteps, high = spread out. Averaged over heads and query positions,
    one value per layer.

    Works on `decoder_xattn` / `encoder_selfattn` / `decoder_selfattn`
    (shape [n_layers, heads, L, S]). Returns one labelled vector per attention
    tensor found, packed as named sub-tensors.
    """
    out_tensors: Dict[str, Any] = {}
    for name in ("decoder_xattn", "encoder_selfattn", "decoder_selfattn"):
        if name not in tensors:
            continue
        a = _as_array(tensors[name])              # [n_layers, heads, L, S]
        if a.ndim != 4:
            continue
        p = np.clip(a, 1e-9, 1.0)
        ent = -(p * np.log(p)).sum(axis=-1)        # [n_layers, heads, L]
        per_layer = ent.mean(axis=(1, 2))          # [n_layers]
        out_tensors[f"{name}_entropy"] = {
            "shape": [int(per_layer.shape[0])],
            "data": [float(v) for v in per_layer],
        }
    return {"tensors": out_tensors}


def activation_norm(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Per-timestep L2 norm ("activity energy") of latent-state tensors.

    For `encoder_out` / `decoder_out` (shape [T, d_model]), the L2 norm across
    the feature dimension gives a compact "how active is the model at each
    timestep" signal — a simple activity meter that pulses with motion.
    """
    out_tensors: Dict[str, Any] = {}
    for name in ("encoder_out", "decoder_out"):
        if name not in tensors:
            continue
        a = _as_array(tensors[name])              # [T, d_model]
        if a.ndim != 2:
            continue
        norms = np.linalg.norm(a, axis=-1)         # [T]
        out_tensors[f"{name}_norm"] = {
            "shape": [int(norms.shape[0])],
            "data": [float(v) for v in norms],
        }
    return {"tensors": out_tensors}


# Registry — reference these names from bridge_channels.yaml `derived:` entries.
TRANSFORMS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "attention_entropy": attention_entropy,
    "activation_norm": activation_norm,
}


def resolve_transform(name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    if name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform '{name}'. Available: {sorted(TRANSFORMS)}"
        )
    return TRANSFORMS[name]
