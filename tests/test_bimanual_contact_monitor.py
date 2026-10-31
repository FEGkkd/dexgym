"""Contact pairing and episode history must remain isolated by environment."""

from dexteroushands_lab.scenes.contact_monitor import (
    ContactHistory, ContactMonitorConfig, classify_contact_pair,
)


def test_contact_pair_classifies_object_and_hand_links():
    bottle = "/World/envs/env_0/Bottle"
    cap = "/World/envs/env_0/Cap"
    left = "/World/envs/env_0/LeftHand/l_index_finger_middle"
    right = "/World/envs/env_0/RightHand/r_thumb_distal"
    ground = "/World/ground/GroundPlane/CollisionPlane"
    assert classify_contact_pair(bottle, left) == "left_bottle"
    assert classify_contact_pair(cap, right) == "right_cap"
    assert classify_contact_pair(cap, left) == "left_cap"
    assert classify_contact_pair(bottle, right) == "right_bottle"
    assert classify_contact_pair(left, right) == "hand_hand"
    assert classify_contact_pair(bottle, ground) == "bottle_ground"
    assert classify_contact_pair(cap, ground) == "cap_ground"
    assert classify_contact_pair(bottle,
                                 "/World/envs/env_1/LeftHand/l_index_finger_middle") is None


def test_history_reset_drops_only_selected_environment_and_requires_new_step():
    history = ContactHistory(2, ContactMonitorConfig(detection_force_n=0.01))
    for env in (0, 1):
        history.record(env, "left_bottle", "l_index_finger_middle", "Bottle", 1.0, 0.002,
                       position=(0, 0, 0.5), separation_m=-0.001)
        history.finish_step(env, 0.002)
    assert history.summary(0)["left_bottle"]["contact_ratio"] == 1.0
    assert (history.summary(0)["left_bottle"]["bodies"]
            ["l_index_finger_middle|Bottle"]["minimum_separation_m"] == -0.001)
    assert history.summary(1)["left_bottle"]["contact_ratio"] == 1.0
    history.reset((0,))
    assert history.summary(0) == {}
    assert history.summary(1)["left_bottle"]["contact_ratio"] == 1.0
    assert not history.is_fresh(0)
    assert history.is_fresh(1)
    history.finish_step(0, 0.002)
    assert history.is_fresh(0)
    assert history.summary(0) == {}
