"""Launch one standalone Isaac fixture and require its explicit result marker."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


FIXTURE = Path(__file__).with_name("smoke_bimanual_scene.py")


def run_fixture(*arguments: str) -> dict:
    command = [sys.executable, "-u", str(FIXTURE), "--headless", *arguments]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    result_lines = [line.split("BIMANUAL_SCENE_RESULT ", 1)[1]
                    for line in process.stdout.splitlines()
                    if line.startswith("BIMANUAL_SCENE_RESULT ")]
    if process.returncode or len(result_lines) != 1 or "BIMANUAL_ERROR" in process.stdout:
        print(process.stdout[-12000:], file=sys.stderr)
        raise RuntimeError(f"Isaac fixture failed: exit={process.returncode}, args={arguments}")
    return json.loads(result_lines[0])
