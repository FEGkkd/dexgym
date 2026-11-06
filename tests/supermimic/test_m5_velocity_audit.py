"""A physical hold-speed audit must distinguish joint velocity from drift."""

import pytest
import torch


def test_substep_velocity_audit_preserves_instantaneous_and_displacement_speeds():
    from dexgym.tasks.marvin_supermimic.velocity_audit import substep_velocity_metrics
    q = torch.tensor([[0.0], [0.001], [0.0], [0.001]])
    qd = torch.tensor([[0.5], [-0.5], [0.5], [-0.5]])
    report = substep_velocity_metrics(q, qd, dt=0.002)
    assert report["sample_count"] == 4
    assert report["instantaneous_abs_max_rad_s"] == pytest.approx(0.5)
    assert report["substep_displacement_abs_max_rad_s"] == pytest.approx(0.5)
    assert report["net_displacement_speed_rad_s"] == pytest.approx(1/6)
    with pytest.raises(ValueError):
        substep_velocity_metrics(q, qd, dt=0)


def test_substep_peak_recorder_keeps_each_policy_step_separate():
    from dexgym.tasks.marvin_supermimic.velocity_audit import SubstepPeakRecorder

    audit = SubstepPeakRecorder(num_envs=2, joints=2, substeps=2)
    audit.begin()
    audit.sample(torch.tensor([[0.1, -0.4], [0.2, 0.3]]))
    audit.sample(torch.tensor([[0.5, 0.1], [0.4, -0.2]]))
    audit.finish(torch.tensor([[0.0, 0.2], [-0.7, 0.1]]))
    audit.begin()
    audit.sample(torch.zeros(2, 2))
    audit.sample(torch.full((2, 2), 0.05))
    audit.finish(torch.zeros(2, 2))
    assert torch.allclose(audit.peaks(), torch.tensor([[0.5, 0.7], [0.05, 0.05]]))
