import pytest

from dexteroushands_lab.tasks.direct.bimanual_bottle.task_contract import (
    CRITIC_FIELDS, critic_force_body_indices,
)


def test_beta2_sensor_frames_do_not_change_critic_width():
    legacy = tuple(f"link_{i}" for i in range(21))
    beta2 = legacy + tuple(f"finger_{i}_tip_sensor_frame" for i in range(5))
    assert critic_force_body_indices(legacy) == tuple(range(21))
    assert critic_force_body_indices(beta2) == tuple(range(21))
    assert CRITIC_FIELDS["left_bottle_forces"] == 21


def test_unknown_contact_body_count_rejected():
    with pytest.raises(ValueError, match="21 hand contact bodies"):
        critic_force_body_indices(("wrist",))
