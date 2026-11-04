import hashlib
from pathlib import Path

from dexgym.assets.catalog import load_manifest


def test_local_assets_unchanged():
    manifest = load_manifest()
    root = Path(__file__).resolve().parents[2]
    for item in manifest["local_files"]:
        path = root / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
