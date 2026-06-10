"""Generic ROS-message -> plain-dict serializers for the viz bridge.

This module is the scalability core of the bridge. Each serializer turns a ROS2
message into a JSON-friendly dict with a *uniform* shape so that every client
(Unity / Unreal / browser) can use a single parser regardless of channel:

    {
        "channel": "<set by the node>",
        "stamp":   <float seconds, set by the node>,
        "shape":   [d0, d1, ...],   # flat-array reshape hint
        "data":    [ ... ],         # flat list of floats
        "labels":  [ ... ],         # optional, per-leading-dim names
    }

or, for messages carrying several named tensors (e.g. NN activations):

    {
        "channel": "...",
        "stamp":   ...,
        "tensors": { "<name>": {"shape": [...], "data": [...]}, ... }
    }

Adding a brand-new *message type* means adding one function here and registering
it in SERIALIZERS. Adding a new *topic* of an already-supported type needs only a
config line (no code) — see bridge_channels.yaml.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _multiarray_shape(layout) -> list:
    """Recover a shape list from a std_msgs MultiArrayLayout, or [] if unset."""
    try:
        return [int(d.size) for d in layout.dim if int(d.size) > 0]
    except Exception:
        return []


def _as_float_list(seq) -> list:
    """Coerce any array-like (python list, array.array, numpy) to a list of floats."""
    return [float(v) for v in seq]


# --------------------------------------------------------------------------- #
# Serializers (each: msg -> dict fragment, the node adds channel + stamp)
# --------------------------------------------------------------------------- #
def serialize_joint_state(msg) -> Dict[str, Any]:
    """sensor_msgs/msg/JointState -> positions with joint-name labels."""
    data = _as_float_list(msg.position)
    return {
        "shape": [len(data)],
        "data": data,
        "labels": list(msg.name),
        "velocity": _as_float_list(msg.velocity) if len(msg.velocity) else [],
    }


def serialize_multiarray(msg) -> Dict[str, Any]:
    """std_msgs/msg/Float32MultiArray or Float64MultiArray -> flat data + shape.

    Shape is recovered from layout.dim when present; otherwise a flat [N].
    """
    data = _as_float_list(msg.data)
    shape = _multiarray_shape(msg.layout)
    if not shape:
        shape = [len(data)]
    return {"shape": shape, "data": data}


def serialize_multiarray_named(msg) -> Dict[str, Any]:
    """Split one flat Float*MultiArray into several named sub-tensors.

    Convention (produced by the inference node when publish_activations:=true):
    each entry in layout.dim describes one logical tensor, where
        dim[i].label  = tensor name (e.g. "encoder_out", "encoder_selfattn")
        dim[i].size   = total number of elements in that tensor (product of its
                        own dims, encoded as "name:d0xd1xd2")
    To keep it simple and self-describing we encode the per-tensor shape in the
    label as "name:d0xd1x..."; sizes are taken from that. data is the tensors
    concatenated in label order.

    Falls back to serialize_multiarray() if the layout doesn't follow this
    convention, so it never throws on an unexpected message.
    """
    data = _as_float_list(msg.data)
    dims = list(getattr(msg.layout, "dim", []))
    tensors: Dict[str, Any] = {}
    offset = 0
    try:
        for d in dims:
            label = str(d.label)
            name, _, shape_str = label.partition(":")
            if shape_str:
                shape = [int(x) for x in shape_str.split("x") if x]
            else:
                shape = [int(d.size)]
            count = 1
            for s in shape:
                count *= s
            tensors[name] = {"shape": shape, "data": data[offset:offset + count]}
            offset += count
        if not tensors or offset == 0:
            return serialize_multiarray(msg)
        return {"tensors": tensors}
    except Exception:
        return serialize_multiarray(msg)


def serialize_zed_skeleton(msg) -> Dict[str, Any]:
    """zed_msgs/msg/ObjectsStamped -> first tracked body's keypoints [K,3].

    Returns an empty frame (shape [0,3]) when no body is present so downstream
    clients always receive a well-formed message.
    """
    objects = list(getattr(msg, "objects", []))
    if not objects:
        return {"shape": [0, 3], "data": [], "object_id": -1}
    obj = objects[0]
    kps = getattr(obj, "skeleton_3d", None)
    keypoints = getattr(kps, "keypoints", []) if kps is not None else []
    data: list = []
    for kp in keypoints:
        kv = getattr(kp, "kp", kp)
        data.extend([float(kv[0]), float(kv[1]), float(kv[2])])
    return {
        "shape": [len(data) // 3, 3],
        "data": data,
        "object_id": int(getattr(obj, "label_id", -1)),
    }


def serialize_generic_numeric(msg) -> Dict[str, Any]:
    """Last-resort fallback: pull any numeric / numeric-sequence fields into a dict.

    Lets a new topic be bridged with no custom serializer, at the cost of a less
    tidy payload. Non-numeric fields are skipped.
    """
    out: Dict[str, Any] = {}
    for field in getattr(msg, "get_fields_and_field_types", lambda: {})().keys():
        try:
            value = getattr(msg, field)
        except Exception:
            continue
        if isinstance(value, bool):
            out[field] = value
        elif isinstance(value, (int, float)):
            out[field] = float(value)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
            out[field] = _as_float_list(value)
    return {"fields": out}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Explicit serializer names usable from config via the `serializer:` key.
SERIALIZERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "joint_state": serialize_joint_state,
    "multiarray": serialize_multiarray,
    "multiarray_named": serialize_multiarray_named,
    "zed_skeleton": serialize_zed_skeleton,
    "generic_numeric": serialize_generic_numeric,
}

# Default serializer chosen automatically from the ROS message `type` string when
# no explicit `serializer:` is given in config.
TYPE_DEFAULTS: Dict[str, str] = {
    "sensor_msgs/msg/JointState": "joint_state",
    "std_msgs/msg/Float32MultiArray": "multiarray",
    "std_msgs/msg/Float64MultiArray": "multiarray",
    "zed_msgs/msg/ObjectsStamped": "zed_skeleton",
}


def resolve_serializer(type_str: str, serializer_name: str | None) -> Callable[[Any], Dict[str, Any]]:
    """Pick a serializer: explicit name wins, else by message type, else generic."""
    if serializer_name:
        if serializer_name not in SERIALIZERS:
            raise KeyError(
                f"Unknown serializer '{serializer_name}'. "
                f"Available: {sorted(SERIALIZERS)}"
            )
        return SERIALIZERS[serializer_name]
    default = TYPE_DEFAULTS.get(type_str)
    if default:
        return SERIALIZERS[default]
    return SERIALIZERS["generic_numeric"]
