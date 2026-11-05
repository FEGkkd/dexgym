"""The aligned trainer must save a usable resume point without Replay state."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "supermimic"))
from train_aligned import _load_resume, _save_checkpoint_atomic  # noqa: E402


def test_atomic_checkpoint_restores_networks_and_optimizers(tmp_path):
    modules = [torch.nn.Linear(2, 2) for _ in range(3)]
    optimizers = [torch.optim.Adam(module.parameters()) for module in modules]
    for module, optimizer in zip(modules, optimizers, strict=True):
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    expected = [[parameter.detach().clone() for parameter in module.parameters()] for module in modules]
    path = tmp_path / "latest.pt"
    _save_checkpoint_atomic(path, *modules, *optimizers, {"iteration": 7})
    assert path.is_file() and not path.with_suffix(".pt.tmp").exists()
    for module in modules:
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
    training, missing = _load_resume(path, *modules, *optimizers, torch.device("cpu"))
    assert training["iteration"] == 7 and missing == []
    for module, target in zip(modules, expected, strict=True):
        for parameter, value in zip(module.parameters(), target, strict=True):
            torch.testing.assert_close(parameter, value)
