import math
from pathlib import Path

import numpy as np

from scripts.environments.audit_phase1b_cap_contact_geometry import (
    box_sdf, cylinder_sdf, read_external_proxy, world_to_cap,
)


def test_proxy_signed_distances_and_rotated_box():
    center = np.zeros(3)
    assert math.isclose(box_sdf(np.zeros(3), center, np.ones(3), 0), -1)
    assert math.isclose(box_sdf(np.array((2., 0., 0.)), center, np.ones(3), 0), 1)
    assert math.isclose(box_sdf(np.array((0., 2.5, 0.)), center,
                                np.array((2., 1., 1.)), math.pi / 2), 0.5)
    assert math.isclose(cylinder_sdf(np.zeros(3), center, 1, 1), -1)
    assert math.isclose(cylinder_sdf(np.array((1.4, 0., 0.)), center, 1, 1), 0.4)
    assert math.isclose(cylinder_sdf(np.array((0., 0., 2.)), center, 1, 1), 1)


def test_contact_point_transforms_to_cap_frame():
    c = math.sqrt(0.5)
    pose = np.array((1., 1., 3., 0., 0., c, c))
    np.testing.assert_allclose(world_to_cap((1., 2., 3.), pose), (1., 0., 0.), atol=1e-12)


def test_audit_reads_authored_cap_collision_shapes():
    usd = Path(__file__).resolve().parents[1] / (
        "source/dexteroushands_lab/assets/bottle/generated/cap.usda")
    walls, extra = read_external_proxy(usd)
    assert len(walls) == 12
    assert math.isclose(extra["top"][1], 0.017)
    assert all(not attrs for attrs in extra["offset_attrs"].values())
