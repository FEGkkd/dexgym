"""Run original checks against their own immutable Git objects."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_design import HERE, REPO, git, read_json


def verify(commit, relative, arguments, docs_only=False):
    args = ["archive", "--format=zip", commit]
    if docs_only:
        args.extend(["--", "docs/s0-5"])
    snapshot = git(*args)
    with TemporaryDirectory(prefix="s05-historical-") as folder:
        checkout = Path(folder)
        with ZipFile(BytesIO(snapshot)) as archive:
            archive.extractall(checkout)
        env = {**os.environ, "GIT_DIR": str(REPO / ".git"), "GIT_WORK_TREE": str(checkout),
               "PYTHONIOENCODING": "utf-8"}
        print(f"Historical Git object {commit}: {relative}", flush=True)
        subprocess.run([sys.executable, "-B", str(checkout / relative), *arguments],
                       cwd=checkout, env=env, check=True)


def main():
    baseline = read_json(HERE / "historical_inputs.json")["reviewed_baseline"]
    verify(baseline, "docs/s0-5/evidence/P00/check_documents.py", ["--require-history"])
    verify("09c1c9e3c63797083a82493e2a40dd8bb4b1db6d",
           "docs/s0-5/evidence/P00_review_fix/test_review_checks.py", [], docs_only=True)


if __name__ == "__main__":
    main()
