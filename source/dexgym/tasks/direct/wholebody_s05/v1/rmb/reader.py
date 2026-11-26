"""Read-only RMB loading with split, fingerprint, and numeric-array checks."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .schema import SCHEMA, SPLITS, TICK_NS, file_sha


@dataclass(frozen=True)
class RMBPartition:
    key: str
    root: str
    split: str
    arrays: dict[str, np.ndarray]


class RobotReferenceBank:
    def __init__(self, directory, *, expected_sha=None, verify=True, require_qualified=True,
                 review_receipt=None, expected_review_sha=None):
        self.directory = Path(directory).resolve()
        path = self.directory / "rmb_manifest.json"
        manifest_sha = file_sha(path)
        if expected_sha is not None and manifest_sha != expected_sha:
            raise ValueError("RMB manifest does not match the requested identity")
        self.manifest = json.loads(path.read_bytes())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unrecognized RMB schema")
        if require_qualified:
            if self.manifest["qualification"] != "READY_FOR_REVIEW" or review_receipt is None or expected_review_sha is None:
                raise ValueError("RMB archive is not eligible for policy training without an independent receipt")
            review_path = Path(review_receipt)
            if file_sha(review_path) != expected_review_sha:
                raise ValueError("RMB review receipt fingerprint mismatch")
            receipt = json.loads(review_path.read_bytes())
            if (receipt.get("gate") != "G-A" or receipt.get("verdict") != "TECHNICAL_PASS"
                    or receipt.get("manifest_sha256") != manifest_sha or receipt.get("independent_reviewer") in
                    (None, "", receipt.get("implementation_agent"))):
                raise ValueError("RMB review does not qualify this bank")
        self.review_verified = bool(require_qualified)
        self.verify = verify
        parts = self.manifest["partitions"]
        keys = [row["key"] for row in parts]
        if len(set(keys)) != len(keys):
            raise ValueError("RMB partition keys overlap")
        roots = []
        for split in SPLITS:
            roots.append({row["root"] for row in parts if row["split"] == split})
        if roots[0] & roots[1]:
            raise ValueError("RMB root split is contaminated")
        self.index = {row["key"]: row for row in parts}

    def partitions(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unsupported RMB split: {split}")
        for key, record in self.index.items():
            if record["split"] == split:
                yield self.read(key, split)

    def read(self, key, split):
        record = self.index[key]
        if record["split"] != split:
            raise ValueError("Requested partition cannot be read from this split")
        directory = (self.directory / record["relative_path"]).resolve()
        if not directory.is_relative_to(self.directory):
            raise ValueError("RMB partition is outside its manifest directory")
        arrays = {}
        for name, descriptor in record["arrays"].items():
            path = (directory / (name + ".npy")).resolve()
            if not path.is_relative_to(directory) or path.parent != directory:
                raise ValueError(f"Unsafe array path: {name}")
            if self.verify and file_sha(path) != descriptor["sha256"]:
                raise ValueError(f"RMB array fingerprint mismatch: {name}")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(data.shape) != descriptor["shape"] or data.dtype.str != descriptor["dtype"]:
                raise ValueError(f"RMB array shape or dtype mismatch: {name}")
            arrays[name] = data
        if np.any(np.diff(arrays["tick_ns"]) != TICK_NS):
            raise ValueError("Noncontinuous time axis inside one planned partition")
        return RMBPartition(key, record["root"], split, arrays)
