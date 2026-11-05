import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "supermimic"))
from train_aligned import _parse_args  # noqa: E402


def test_training_options_keep_legacy_default_and_expose_paper_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_aligned", "--num-envs", "2", "--iterations", "1"])
    default = _parse_args()
    assert default.normalization_mode == "legacy"
    assert default.termination_mode == "learned"
    assert (default.critic_epochs, default.critic_minibatches) == (1, 1)
    monkeypatch.setattr(sys, "argv", ["train_aligned", "--num-envs", "2", "--iterations", "1",
                                      "--normalization-mode", "running", "--termination-mode", "analytic",
                                      "--critic-epochs", "5", "--critic-minibatches", "4"])
    selected = _parse_args()
    assert selected.normalization_mode == "running"
    assert selected.termination_mode == "analytic"
    assert (selected.critic_epochs, selected.critic_minibatches) == (5, 4)
