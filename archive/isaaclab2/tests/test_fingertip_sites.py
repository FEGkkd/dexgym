import math
from types import SimpleNamespace

import pytest
import torch

from dexteroushands_lab.scenes.fingertip_sites import (
    FingertipSiteTracker, bottle_axis_radial_distance,
)


def test_radial_distance_uses_tilted_bottle_local_axis():
    qy = math.sin(math.pi / 4)
    radius, axial = bottle_axis_radial_distance(
        (0.1, 0.035, 0.5), (0, 0, 0.5, 0, qy, 0, qy))
    assert radius == pytest.approx(0.035)
    assert axial == pytest.approx(0.1)


def test_tracker_requires_tip_frames_and_reads_live_pose_buffer():
    names = [f"l_{d}_tip" for d in
             ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
    poses = torch.zeros((1, 5, 7))
    poses[0, :, 0] = 0.04
    hand = SimpleNamespace(body_names=names,
                           data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=poses)))
    spec = SimpleNamespace(variant="beta1", side="left",
                           fingertip_bodies=tuple(x.replace("_tip", "_distal") for x in names))
    tracker = FingertipSiteTracker(hand, spec)
    first = tracker.observe(0, (0, 0, 0.5, 0, 0, 0, 1), 0.035)
    assert first["thumb"]["radial_distance_to_bottle_axis_m"] == pytest.approx(0.04)
    poses[0, 0, 0] = 0.06
    second = tracker.observe(0, (0, 0, 0.5, 0, 0, 0, 1), 0.035)
    assert second["thumb"]["radial_distance_to_bottle_axis_m"] == pytest.approx(0.06)
    with pytest.raises(RuntimeError, match="Missing live"):
        FingertipSiteTracker(SimpleNamespace(body_names=names[:4]), spec)


def test_beta2_tracker_uses_tip_frames_alongside_sensor_frames():
    names = [f"r_{d}_tip" for d in
             ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
    body_names = [name for tip in names for name in (tip, tip + "_sensor_frame")]
    poses = torch.zeros((1, len(body_names), 7))
    poses[0, ::2, 0] = 0.04
    hand = SimpleNamespace(body_names=body_names,
                           data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=poses)))
    spec = SimpleNamespace(variant="beta2", side="right",
                           fingertip_bodies=tuple(x.replace("_tip", "_distal") for x in names))
    tracker = FingertipSiteTracker(hand, spec)
    assert tracker.names == tuple(names)
    assert tracker.observe(0, (0, 0, 0.5, 0, 0, 0, 1), 0.035)["thumb"][
        "radial_distance_to_bottle_axis_m"] == pytest.approx(0.04)
