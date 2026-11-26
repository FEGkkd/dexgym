"""Replay v2.0.0 checks in a temporary checkout of their exact Git baseline."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_design import ARCHIVE, git, legacy


def main():
    snapshot = git("archive", "--format=zip", ARCHIVE)
    with TemporaryDirectory(prefix="s05-v200-baseline-") as directory:
        root = Path(directory)
        with ZipFile(BytesIO(snapshot)) as archive:
            archive.extractall(root)
        env = {**os.environ, "GIT_DIR": str(legacy.REPO / ".git"), "GIT_WORK_TREE": str(root),
               "PYTHONIOENCODING": "utf-8"}
        for name in ("check_design.py", "test_check_design.py"):
            command = root / "docs/s0-5/evidence/P00_V2" / name
            print(f"Original candidate {ARCHIVE}: {name}", flush=True)
            subprocess.run([sys.executable, "-B", str(command)], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
