"""Motion Library manifests and experiment reports."""

import json
from pathlib import Path

import numpy as np


def clip_paths(manifest):
    paths = [Path(path) for path in json.loads(manifest.read_text())]
    if not paths or len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        raise ValueError(f"Expected a list of distinct existing clip files in {manifest}")
    return paths


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_episode(path):
    with np.load(path, allow_pickle=False) as clip:
        return str(clip["source_episode"].item())
