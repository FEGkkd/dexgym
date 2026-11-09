"""Pack model resources into MuJoCo virtual file systems."""

from pathlib import Path


def update_assets(assets, path, meshdir=None, glob="*", recursive=False):
    for item in Path(path).glob(glob):
        if item.is_file():
            assets[f"{meshdir}/{item.name}" if meshdir else item.name] = item.read_bytes()
        elif recursive:
            update_assets(assets, item, meshdir, glob, recursive)
