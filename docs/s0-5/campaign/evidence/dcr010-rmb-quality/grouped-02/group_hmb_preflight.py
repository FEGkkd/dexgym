from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import numpy as np
from build_offline_rmb_diagnostic import (
    DOCS, RAW_HMB_FIELDS, ROOT_FIVE, blank, file_sha, publish_partition, train_reader,
)
roots, reader, sha = train_reader(Path(sys.argv[1]), DOCS, None)
root, = (x for x in roots if x["root"] == ROOT_FIVE)
assert root["split"] == "Train"
paths = root["hmb_partitions"]
total = 0
for part in paths:
    data, meta, sources, _ = reader.read(part["key"])
    fields = blank(data, grouped=True)
    keys = {"hmb_" + name: name for name in RAW_HMB_FIELDS}
    keys["hmb_source_index_full"] = "source_index"
    for name, field in fields.items():
        if name in keys:
            expected = data[keys[name]]
            assert field.dtype == expected.dtype and field.shape == expected.shape, name
            assert field.tobytes() == expected.tobytes(), name
        elif isinstance(field, dict):
            for geom, array in field.items():
                assert np.isfinite(array).all(), (name, geom)
        else:
            assert field.dtype.kind in "biuf" and np.isfinite(field).all(), name
    provenance = len(json.dumps((meta, sources), allow_nan=False))
    if part is paths[0] or part is paths[-1]:
        with TemporaryDirectory(prefix="grouped-rmb-schema-preflight-") as scratch:
            output = Path(scratch) / "partition"
            published = publish_partition(output, part["key"], fields)
            for name, raw in keys.items():
                expected = data[raw]
                path = output / (name + ".npy")
                indexed = published["arrays"][name]
                stored = np.load(path, allow_pickle=False)
                assert indexed["raw_diagnostic_only"] is True, name
                assert indexed["sha256"] == file_sha(path), name
                assert stored.dtype == expected.dtype and stored.shape == expected.shape, name
                assert stored.tobytes() == expected.tobytes(), name
            print("DISK_LOSSLESS_PASS", part["key"], len(keys), flush=True)
    rows = len(data["tick_ns"])
    total += rows
    print("PARTITION_PASS", part["key"], rows, provenance, flush=True)
assert len(paths) == 13 and total == 4844, (len(paths), total)
print("ALL_SIGNED_PARTITIONS_PASS", len(paths), total, sha, flush=True)
