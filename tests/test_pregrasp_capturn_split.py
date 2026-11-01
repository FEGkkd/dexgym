from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_split import split_group_indices


@dataclass
class Row:
    metadata: dict


def test_split_keeps_all_local_variants_of_one_source_in_one_partition():
    rows = [Row({"right_natural_source_seed": source})
            for source in (10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15)]
    train = split_group_indices(rows, key="right_natural_source_seed", evaluation=False)
    evaluation = split_group_indices(rows, key="right_natural_source_seed", evaluation=True)
    assert set(train).isdisjoint(evaluation)
    assert set(train) | set(evaluation) == set(range(len(rows)))
    train_sources = {rows[i].metadata["right_natural_source_seed"] for i in train}
    eval_sources = {rows[i].metadata["right_natural_source_seed"] for i in evaluation}
    assert train_sources.isdisjoint(eval_sources)


def test_frozen_global_sources_do_not_leak_across_reset_levels():
    a = [Row({"right_natural_source_seed": x}) for x in (83, 860, 1485, 1625, 1779, 2261)]
    b = [Row({"right_natural_source_seed": x}) for x in (83, 860, 1485, 1625, 1779, 1881, 2261)]
    evaluation_sources = {1625, 1779, 1881, 2261}
    a_eval = split_group_indices(a, key="right_natural_source_seed",
                                 evaluation=True, evaluation_groups=evaluation_sources)
    b_train = split_group_indices(b, key="right_natural_source_seed",
                                  evaluation=False, evaluation_groups=evaluation_sources)
    assert {a[i].metadata["right_natural_source_seed"] for i in a_eval}.isdisjoint(
        {b[i].metadata["right_natural_source_seed"] for i in b_train})


def test_checked_in_bank_and_manifest_have_global_source_separation():
    root = Path(__file__).resolve().parents[1]
    bank = root / "docs/assets/phase1b_pregrasp_bank.json"
    manifest = json.loads((root / "docs/assets/phase1b_pregrasp_split.json").read_text())
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == manifest["bank_sha256"]
    train = set()
    evaluation = set()
    for split in manifest["splits"].values():
        train.update(split["train_natural_source_seeds"])
        evaluation.update(split["evaluation_natural_source_seeds"])
    assert train.isdisjoint(evaluation)
    assert evaluation == set(manifest["evaluation_natural_source_seeds_global"])
