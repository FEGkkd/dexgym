"""Fetch the pinned CMU BVH mirror with sparse Subject selection for reproducible pilots."""

import argparse
import subprocess
from pathlib import Path

REMOTE = "https://github.com/una-dinosauria/cmu-mocap.git"
DEFAULT_ROOT = Path("/home/ubuntu/dexgym_data/cmu/cmu-mocap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=["141"])
    parser.add_argument("--commit", default="09a07f54f3bbb58797325f009282d0b2048a2871")
    args = parser.parse_args()
    repo = args.repo.resolve()
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REMOTE, str(repo)], check=True
        )
    remote = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
    if remote != REMOTE:
        raise ValueError(f"Unexpected CMU remote: {remote}")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.commit:
        raise ValueError(f"CMU commit changed: {commit} != {args.commit}")
    paths = [f"data/{subject.zfill(3)}" for subject in args.subjects]
    subprocess.run(
        ["git", "-c", "http.version=HTTP/1.1", "-C", str(repo), "sparse-checkout", "set", *paths], check=True
    )
    files = sorted(repo.glob("data/*/*.bvh"))
    print(f"CMU commit {commit}, materialized {len(files)} BVH files in {repo}")


if __name__ == "__main__":
    main()
