from dexteroushands_lab.tasks.direct.bimanual_bottle.reset_sampler import (
    split_bank_indices, split_pregrasp_indices)


def test_eval_samples_are_disjoint_from_train_samples():
    train_stable, train_pre = split_bank_indices(4, 32, evaluation=False)
    eval_stable, eval_pre = split_bank_indices(4, 32, evaluation=True)
    assert not (set(train_stable) & set(eval_stable))
    assert not (set(train_pre) & set(eval_pre))
    assert len(train_stable) == 3 and len(eval_stable) == 1
    assert len(train_pre) == 24 and len(eval_pre) == 8


def test_one_row_external_stable_eval_keeps_held_out_pregrasp_split():
    assert split_pregrasp_indices(32, evaluation=True) == tuple(range(24, 32))
    assert split_pregrasp_indices(32, evaluation=False) == tuple(range(24))
