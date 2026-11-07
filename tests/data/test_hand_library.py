"""Generated library must preserve sequence timing and robot reachability."""

import json
from pathlib import Path

import numpy as np
import pytest
from dexgym.data.library import build_library, sample_indices
from dexgym.data.wuji_retarget import RightHandFK

ROOT = Path(__file__).resolve().parents[2]
SDK = Path("/home/ubuntu/teleop_ws/third_party/wuji_c_api/lib/libwuji_sdk_c.so")
RIGHT_XML = Path(
    "/home/ubuntu/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml"
)
SAMPLE = ROOT / "data/grab/s1/airplane_fly_1.npz"


def test_sample_indices_keeps_exact_thirty_hertz_timing():
    np.testing.assert_array_equal(sample_indices(9, 120.0, 30.0), [0, 4, 8])
    with pytest.raises(ValueError, match="integer"):
        sample_indices(10, 100.0, 30.0)


@pytest.mark.skipif(not all(p.is_file() for p in (SAMPLE, SDK, RIGHT_XML)), reason="local inputs absent")
def test_one_motion_library_roundtrips_reachable_targets(tmp_path):
    manifest = build_library(
        grab_root=ROOT / "data/grab",
        data_root=ROOT / "data",
        mano_path=ROOT / "data/models/mano_v1_2/MANO_RIGHT.pkl",
        sdk_path=SDK,
        right_xml=RIGHT_XML,
        output_dir=tmp_path,
        selected=[SAMPLE],
    )
    assert manifest["accepted"] == 1
    assert manifest["rejected"] == 0
    entry = manifest["motions"][0]
    assert entry["split"] == "train"
    assert entry["frames"] == 279
    path = tmp_path / entry["file"]
    with np.load(path, allow_pickle=False) as motion:
        q = motion["q_ref"]
        tips = motion["tip_ref"]
        assert q.shape == (279, 20)
        assert tips.shape == (279, 5, 3)
        assert float(motion["fps"]) == 30.0
        np.testing.assert_array_equal(motion["source_indices"][:3], [0, 4, 8])
    fk = RightHandFK(RIGHT_XML)
    for frame in (0, 100, 278):
        np.testing.assert_allclose(tips[frame], fk.tips(q[frame]), atol=1e-6)
    assert json.loads((tmp_path / "manifest.json").read_text())["accepted"] == 1
