import hashlib
from pathlib import Path

from dexteroushands_lab.tasks.direct.bimanual_bottle.experiment_manifest import snapshot_files


def test_training_snapshot_preserves_exact_source_bytes(tmp_path):
    root = tmp_path / "repo"
    log_dir = tmp_path / "logs"
    relative = Path("task/model.py")
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x = 1\n")
    hashes = snapshot_files(log_dir, root, [relative])
    assert hashes[relative.as_posix()] == hashlib.sha256(b"x = 1\n").hexdigest()
    assert (log_dir / "source_snapshot" / relative).read_bytes() == source.read_bytes()
