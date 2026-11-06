"""The hand-contact override must retain the original robot physics."""

import json
import subprocess
import sys
from pathlib import Path

from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_USD

ROOT = Path(__file__).resolve().parents[2]


def run_script(name: str, *args: str) -> None:
    # Isaac Sim's Python may need site packages appended *after* importing its
    # bundled NumPy. Preserve the parent test runner's path in the child.
    bootstrap = (
        "import json, numpy, runpy, sys; "
        "sys.path.extend(json.loads(sys.argv.pop(1))); "
        "script = sys.argv.pop(1); sys.argv[0] = script; "
        "runpy.run_path(script, run_name='__main__')"
    )
    subprocess.run(
        [sys.executable, "-c", bootstrap, json.dumps(sys.path),
         str(ROOT / "scripts" / "assets" / name), *args],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def test_packaged_no_hand_asset_preserves_mass_inertia_and_mounts(tmp_path):
    output = tmp_path / "structure.json"
    run_script("validate_marvin_hand_contacts.py", "--output", str(output))
    report = json.loads(output.read_text())
    assert report["pass"]
    assert (report["rigid_bodies"], report["revolute_joints"], report["fixed_joints"]) == (74, 54, 20)
    assert (report["hand_collision_shapes_disabled"], report["arm_collision_shapes_enabled"]) == (54, 33)


def test_no_hand_override_can_be_regenerated(tmp_path):
    output = tmp_path / "no_hand_contacts.usda"
    result = tmp_path / "structure.json"
    run_script("disable_marvin_hand_contacts.py", "--base-usd", str(PACKAGED_USD), "--output", str(output))
    run_script(
        "validate_marvin_hand_contacts.py",
        "--full-usd",
        str(PACKAGED_USD),
        "--no-hand-usd",
        str(output),
        "--output",
        str(result),
    )
    assert json.loads(result.read_text())["pass"]
