import torch


def test_baseline_dispatch_ignores_checkpoint_actor():
    from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.action_math import base_evaluation_action

    calls = []

    def actor(obs):
        calls.append(True)
        return torch.full((2, 52), 0.75, device=obs.device)

    obs = torch.zeros((2, 277))
    zero = base_evaluation_action("zero", actor, obs)
    random = base_evaluation_action("random", actor, obs)
    no_hands = base_evaluation_action("no_hands", actor, obs)
    learned = base_evaluation_action("learned", actor, obs)
    assert torch.count_nonzero(zero) == 0
    assert torch.count_nonzero(no_hands) == 0
    assert random.shape == (2, 52)
    assert not torch.equal(random, learned)
    assert torch.all(learned == 0.75)
    assert calls == [True]
