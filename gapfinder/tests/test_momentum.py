from __future__ import annotations

from gapfinder.graph.momentum import yearly_concept_share


def test_yearly_concept_momentum_uses_shares_not_stratified_volume() -> None:
    original = [
        {"year": 2020, "concepts": ["GNN"]},
        {"year": 2020, "concepts": ["Other"]},
        {"year": 2021, "concepts": ["GNN"]},
        {"year": 2021, "concepts": ["Other"]},
    ]
    duplicated = original + original
    first = yearly_concept_share(
        original, "gnn", bootstrap_resamples=1000, seed=5
    )
    second = yearly_concept_share(
        duplicated, "gnn", bootstrap_resamples=1000, seed=5
    )

    assert first[2020]["share"] == second[2020]["share"] == 0.5
    assert first[2021]["share"] == second[2021]["share"] == 0.5
    assert first[2020]["papers"] == 2
    assert second[2020]["papers"] == 4
    assert first[2020]["ci95_low"] <= 0.5 <= first[2020]["ci95_high"]
