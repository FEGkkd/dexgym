import json
import pytest

from dexgym.algorithms.supermimic.core.config import CoreExperimentConfig


@pytest.mark.parametrize("name,goal_task", [
    ("m43_tracking_smoke.json", "marvin_motion_tracking"),
    ("m43_reach_smoke.json", "marvin_dual_tcp_reach")])
def test_two_task_configs_share_algorithm_and_select_distinct_task(name, goal_task):
    config = CoreExperimentConfig.from_json("configs/supermimic/" + name)
    assert config.task_id == goal_task
    assert config.robot_id == "marvin_arm_only"
    assert config.trainer.world_window == 8
    assert config.trainer.max_imagination_horizon == 32
    assert config.trust.metric == "normalized"


def test_config_rejects_missing_layer(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"algorithm": {}}))
    with pytest.raises(ValueError, match="algorithm, robot, task"):
        CoreExperimentConfig.from_json(path)
