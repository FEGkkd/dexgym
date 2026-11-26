"""Synthetic source splits and bounded Replay history controls."""
import torch
import pytest

from dexgym.tasks.direct.wholebody_s05.v1.learning.replay import CausalReplayBuffer


def record(tick, generation=0, *, done=(False, False), valid=(True, True), role=(True, False),
           source=(11, 22)):
    values = torch.tensor([[tick + 1.0, tick + 1.5], [tick + 101.0, tick + 101.5]])
    return {"obs": values, "source_id": torch.tensor(source), "partition_id": torch.tensor([1, 2]),
            "episode_id": torch.tensor([7, 8]), "generation_id": torch.tensor([generation] * 2),
            "clock_tick": torch.tensor([tick] * 2), "done": torch.tensor(done),
            "valid": torch.tensor(valid), "source_train": torch.tensor(role),
            "terminated": torch.tensor(done), "safety_failure": torch.tensor(done)}


@pytest.fixture
def buffer():
    return CausalReplayBuffer(5, 2, "cpu", 3, 1, train_root_ids={11}, holdout_root_ids={22})


def test_current_and_past_equivalent_since_latest_boundary(buffer):
    for tick in range(5):
        buffer.add(record(tick, generation=int(tick >= 3)))
    state = buffer.sample(64, generator=torch.Generator().manual_seed(49))
    expected = {
        0: [0, 0, 0, 0, 1, 1.5], 1: [0, 0, 1, 1.5, 2, 2.5],
        2: [1, 1.5, 2, 2.5, 3, 3.5], 3: [0, 0, 0, 0, 4, 4.5],
        4: [0, 0, 4, 4.5, 5, 5.5],
    }
    for tick, vector in zip(state["clock_tick"][:, 0].tolist(), state["obs"][:, 0].tolist()):
        assert vector == expected[tick]
    assert buffer.data["obs"].shape == (5, 2, 2)
    assert "source_id" not in state and "partition_id" not in state


def test_source_role_cannot_cross_lane(buffer):
    with pytest.raises(ValueError, match="frozen split"):
        buffer.add(record(0, source=(22, 11)))
    with pytest.raises(ValueError, match="frozen split"):
        buffer.add(record(0, role=(False, True)))


def test_wrapped_ring_never_crosses_terminal_and_preserves_failure(buffer):
    buffer = CausalReplayBuffer(7, 2, "cpu", 3, 1, train_root_ids={11}, holdout_root_ids={22})
    for tick in range(8):
        buffer.add(record(tick, done=(tick == 5, False), valid=(tick != 6, True)))
    output = buffer.sample(64, 2, failure_frac=0.5, generator=torch.Generator().manual_seed(49))
    pairs = {tuple(x) for x in output["clock_tick"].tolist()}
    assert pairs == {(3, 4), (4, 5)}
    assert output["safety_failure"][:, 1].any()
    held = buffer.sample(64, window=2, holdout=True)
    assert held["env"].eq(1).all()
    assert held["obs"][..., -2].ge(104).all()
    assert (output["importance"] > 0).all()


def test_wrapped_ring_never_zero_pads_an_evicted_live_prefix(buffer):
    for tick in range(8):
        buffer.add(record(tick))
    sample = buffer.sample(100, generator=torch.Generator().manual_seed(49))
    assert sample["clock_tick"].min().item() == 5
    assert not sample["obs"].eq(0).any()
    with pytest.raises(ValueError, match="lacks real causal predecessors"):
        buffer._history(torch.tensor([0]), torch.tensor([0]))


def test_future_replacement_does_not_change_a_past_builder_row(buffer):
    for tick in range(4):
        buffer.add(record(tick))
    row = torch.tensor([2])
    lane = torch.tensor([0])
    original = buffer._history(row, lane)
    buffer.data["obs"][3, 0].fill_(123.0)
    assert torch.equal(original, buffer._history(row, lane))


def test_imagination_starts_reject_terminated_and_externally_ineligible(buffer):
    for tick in range(4):
        transition = record(tick, done=(tick == 3, False))
        transition["imagination_valid"] = torch.tensor([tick >= 2, True])
        buffer.add(transition)
    chosen = buffer.sample_starts(40)
    assert chosen["clock_tick"].eq(2).all()


def test_schema_changes_refused(buffer):
    buffer.add(record(1))
    row = record(2)
    del row["clock_tick"]
    with pytest.raises(ValueError, match="schema"):
        buffer.add(row)
