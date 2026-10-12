from __future__ import annotations

import math

import numpy as np
import pytest

from gapfinder.graph.build import build_cooccurrence_tensor
from gapfinder.graph.sparsity import (
    benjamini_hochberg,
    expected_count,
    expected_pair_count,
    poisson_lower_tail,
    screen_pairwise_sparse_cells,
    screen_sparse_cells,
)
from gapfinder.graph.topics import kmeans_topic_map


def test_hand_computed_expected_count_and_poisson_lower_tail() -> None:
    # N=20, p(task)=10/20, p(method)=8/20, p(dataset)=10/20:
    # lambda = 20 * 0.5 * 0.4 * 0.5 = 2.
    mean = expected_count(20, 10, 8, 10)
    assert mean == pytest.approx(2.0)
    assert poisson_lower_tail(0, mean) == pytest.approx(math.exp(-2))
    assert poisson_lower_tail(1, mean) == pytest.approx(3 * math.exp(-2))


def test_hand_computed_pair_expected_count_and_pmi() -> None:
    # N_pair=10, left marginal=5, right marginal=4 gives lambda=2.
    assert expected_pair_count(10, 5, 4) == pytest.approx(2.0)
    assert math.log(5 / expected_pair_count(10, 5, 6)) == pytest.approx(
        math.log(5 / 3)
    )


def test_benjamini_hochberg_adjustment_by_hand() -> None:
    # The reverse cumulative minimum changes .045 to .04 at the second rank.
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04]
    )


def test_tensor_uses_only_included_records_and_tests_implicit_zero_cells() -> None:
    records = []
    for index in range(20):
        records.append(
            {
                "paper_id": f"paper-{index}",
                "claims": {
                    "include_in_counts": True,
                    "task": {"value": "Task A" if index < 10 else "Task B"},
                    "method": {"value": "Method A" if index < 8 else "Method B"},
                    "dataset": {"value": "Data A" if index < 10 else "Data B"},
                },
            }
        )
    records.append(
        {
            "paper_id": "excluded-paper",
            "claims": {
                "include_in_counts": False,
                "task": {"value": "Task A"},
                "method": {"value": "Method A"},
                "dataset": {"value": "Data A"},
            },
        }
    )
    canonical = {
        value: value
        for value in ("Task A", "Task B", "Method A", "Method B", "Data A", "Data B")
    }

    tensor = build_cooccurrence_tensor(records, canonical)
    result = screen_sparse_cells(tensor, min_expected=2)

    assert tensor.included_records == 20
    assert tensor.complete_records == 20
    assert tensor.cell_counts[("Task A", "Method A", "Data A")] == 8
    assert result["tested_cells"] == 8
    observed_cell = next(
        row
        for row in result["tested_cell_results"]
        if (row["task"], row["method"], row["dataset"])
        == ("Task A", "Method A", "Data A")
    )
    assert observed_cell["observed_count"] == 8
    assert observed_cell["expected_count"] == pytest.approx(2)
    assert observed_cell["p_value"] == pytest.approx(0.9997625527)
    assert not result["candidates_q_0_05"]


def test_field_specific_map_and_posthoc_dataset_rejection() -> None:
    records = [
        {
            "paper_id": "paper-1",
            "claims": {
                "include_in_counts": True,
                "task": {"value": "shared label"},
                "method": {"value": "shared label"},
                "dataset": {"value": "Cora"},
            },
        },
        {
            "paper_id": "paper-2",
            "claims": {
                "include_in_counts": True,
                "task": {"value": "shared label"},
                "method": {"value": "shared label"},
                "dataset": {"value": "14 real-world graph datasets"},
            },
        },
    ]
    canonical = {
        "schema_version": 2,
        "fields": {
            "task": {"shared label": "task label"},
            "method": {"shared label": "method label"},
            "dataset": {"Cora": "Cora"},
        },
    }

    tensor = build_cooccurrence_tensor(records, canonical)

    assert tensor.marginal_counts["task"] == {"task label": 2}
    assert tensor.marginal_counts["method"] == {"method label": 2}
    assert tensor.marginal_counts["dataset"] == {"Cora": 1}
    assert tensor.specificity_rejections == {"dataset": 1}
    assert tensor.complete_records == 1


def test_pairwise_screen_uses_pair_complete_population_and_reports_pmi() -> None:
    records = []
    for index in range(5):
        records.append(
            {
                "paper_id": f"task-a-method-b-{index}",
                "claims": {
                    "include_in_counts": True,
                    "task": {"value": "Task A"},
                    "method": {"value": "Method B"},
                    "dataset": {"value": None},
                },
            }
        )
    for index in range(5):
        records.append(
            {
                "paper_id": f"task-b-{index}",
                "claims": {
                    "include_in_counts": True,
                    "task": {"value": "Task B"},
                    "method": {"value": "Method A" if index < 4 else "Method B"},
                    "dataset": {"value": None},
                },
            }
        )
    canonical = {
        value: value
        for value in ("Task A", "Task B", "Method A", "Method B")
    }

    tensor = build_cooccurrence_tensor(records, canonical)
    result = screen_pairwise_sparse_cells(tensor, min_expected=2)
    task_method = result["pairwise_results"]["task_x_method"]
    zero_cell = next(
        row
        for row in task_method["tested_cell_results"]
        if row["left_entity"] == "Task A"
        and row["right_entity"] == "Method A"
    )
    observed_cell = next(
        row
        for row in task_method["observed_cell_pmi_results"]
        if row["left_entity"] == "Task A"
        and row["right_entity"] == "Method B"
    )

    assert tensor.included_records == 10
    assert tensor.complete_records == 0
    assert task_method["pair_complete_papers"] == 10
    assert task_method["observed_non_empty_cells"] == 3
    assert task_method["tested_cells_expected_at_least_minimum"] == 4
    assert zero_cell["expected_count"] == pytest.approx(2.0)
    assert zero_cell["observed_count"] == 0
    assert zero_cell["p_value"] == pytest.approx(math.exp(-2))
    assert observed_cell["expected_count"] == pytest.approx(3.0)
    assert observed_cell["observed_count"] == 5
    assert observed_cell["pmi"] == pytest.approx(math.log(5 / 3))


def test_kmeans_topic_map_selects_valid_k_and_returns_cluster_labels() -> None:
    vectors = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
    ]
    mapping, report = kmeans_topic_map(
        ["A", "A alias", "B", "B alias"],
        np.asarray(vectors, dtype=float),
        k_candidates=(2, 3),
        silhouette_sample_size=4,
    )

    assert report["selected_k"] in (2, 3)
    assert len(set(mapping.values())) == report["selected_k"]
    assert mapping["A"] == mapping["A alias"]
    assert mapping["B"] == mapping["B alias"]
