"""Small serialization primitives for the mmap-based, per-partition RMB."""
import hashlib
import json

import numpy as np


SPLITS = ("Train", "Validation")
SCHEMA = "s05.rmb.v1"
TICK_NS = 20_000_000
GEOMETRY_SHAPES = {
    "wrist_xyz": (2, 3),
    "wrist_xyzw": (2, 4),
    "local_landmark_xyz": (2, 5, 4, 3),
    "world_tip_xyz": (2, 5, 3),
    "pinch_xyz": (2, 4, 3),
    "cross_xyz": (5, 5, 3),
    "relative_wrist_xyz": (3,),
    "relative_wrist_xyzw": (4,),
}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, allow_nan=False, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def empty_geometry(length):
    geometry = {}
    for key, shape in GEOMETRY_SHAPES.items():
        geometry[key] = np.zeros((length, *shape), dtype="<f4")
        if key.endswith("xyzw"):
            geometry[key][..., 3] = 1
        geometry[f"{key}_mask"] = np.zeros((length, *shape[:-1]), dtype=bool)
    return geometry


def put_geometry(geometry, row, values):
    for key in GEOMETRY_SHAPES:
        geometry[key][row] = values[key]
        geometry[f"{key}_mask"][row] = True


def add_arrays(destination, prefix, arrays):
    destination.update({f"{prefix}_{key}": value for key, value in arrays.items()})


def write_partition(directory, arrays):
    directory.mkdir(exist_ok=False, parents=True)
    index = {}
    for key, value in sorted(arrays.items()):
        data = np.asarray(value)
        if data.dtype.hasobject or data.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported dtype for {key}: {data.dtype}")
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite field {key}: use masks for missing model values")
        path = directory / f"{key}.npy"
        np.save(path, data.astype(data.dtype.newbyteorder("<"), copy=False), allow_pickle=False)
        index[key] = {"sha256": file_sha(path), "shape": list(data.shape), "dtype": data.dtype.newbyteorder("<").str}
    return index
