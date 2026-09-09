import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from dexteroushands_lab.assets.wuji_hand import WUJI_ISAAC_BODY_NAMES
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.contact import WujiContactGroups


def test_wuji_contact_groups_separate_proximal_distal_and_tip():
    groups = WujiContactGroups.from_body_names(WUJI_ISAAC_BODY_NAMES)
    assert len(groups.palm_proximal) == 11
    assert len(groups.distal) == 10
    assert len(groups.fingertips) == 5
    assert groups.all_hand_bodies == tuple(range(26))
