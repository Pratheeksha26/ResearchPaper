from __future__ import annotations

import bisect
import math
from typing import Any

from gapfinder.graph.build import CooccurrenceTensor

PAIR_AXES = (("task", "method"), ("method", "dataset"), ("task", "dataset"))


def expected_count(
    population: int, task_count: int, method_count: int, dataset_count: int
) -> float:
    """Compute the three-way independent-marginal expected count."""
    if population < 0 or min(task_count, method_count, dataset_count) < 0:
        raise ValueError("Population and marginal counts must be nonnegative.")
    if max(task_count, method_count, dataset_count) > population:
        raise ValueError("A marginal count cannot exceed the population.")
    if population == 0:
        return 0.0
    return task_count * method_count * dataset_count / (population * population)


def expected_pair_count(population: int, left_count: int, right_count: int) -> float:
    """Compute a pairwise independence expected count within pair-complete papers."""
    if population < 0 or min(left_count, right_count) < 0:
        raise ValueError("Population and marginal counts must be nonnegative.")
    if max(left_count, right_count) > population:
        raise ValueError("A pair marginal count cannot exceed its pair population.")
    if population == 0:
        return 0.0
    return left_count * right_count / population


def poisson_lower_tail(observed: int, mean: float) -> float:
    """Return P(X <= observed) for a Poisson random variable with the given mean."""
    if observed < 0:
        raise ValueError("Observed count must be nonnegative.")
    if not math.isfinite(mean) or mean < 0:
        raise ValueError("Poisson mean must be finite and nonnegative.")
    if mean == 0:
        return 1.0
    if observed == 0:
        return math.exp(-mean)

    log_terms = [
        -mean + index * math.log(mean) - math.lgamma(index + 1)
        for index in range(observed + 1)
    ]
    largest = max(log_terms)
    log_probability = largest + math.log(
        math.fsum(math.exp(value - largest) for value in log_terms)
    )
    return min(1.0, math.exp(log_probability))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted q-values in the original p-value order."""
    for value in p_values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite and between zero and one.")
    count = len(p_values)
    if count == 0:
        return []

    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running_min = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running_min = min(running_min, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running_min)
    return adjusted


def screen_sparse_cells(
    tensor: CooccurrenceTensor, *, min_expected: float = 2.0
) -> dict[str, Any]:
    """Test three-way cells with sufficient expectation and BH-correct them."""
    if not math.isfinite(min_expected) or min_expected < 0:
        raise ValueError("min_expected must be finite and nonnegative.")
    population = tensor.included_records
    if population == 0:
        return {
            "population": 0,
            "min_expected": min_expected,
            "tested_cells": 0,
            "candidates_q_0_05": [],
            "candidates_q_0_10": [],
            "tested_cell_results": [],
        }

    dataset_counts = sorted(
        (count, entity)
        for entity, count in tensor.marginal_counts["dataset"].items()
    )
    sorted_dataset_frequencies = [count for count, _ in dataset_counts]
    rows: list[dict[str, Any]] = []

    for task, task_count in tensor.marginal_counts["task"].items():
        for method, method_count in tensor.marginal_counts["method"].items():
            if task_count == 0 or method_count == 0:
                continue
            minimum_dataset_count = (
                min_expected * population * population
                / (task_count * method_count)
            )
            start = bisect.bisect_left(
                sorted_dataset_frequencies, minimum_dataset_count
            )
            for dataset_count, dataset in dataset_counts[start:]:
                mean = expected_count(
                    population, task_count, method_count, dataset_count
                )
                observed = tensor.cell_counts.get((task, method, dataset), 0)
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "dataset": dataset,
                        "observed_count": observed,
                        "expected_count": mean,
                        "p_value": poisson_lower_tail(observed, mean),
                    }
                )

    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, q_value in zip(rows, adjusted, strict=True):
        row["q_value"] = q_value
    rows.sort(
        key=lambda row: (
            row["q_value"],
            row["p_value"],
            -row["expected_count"],
            row["task"].casefold(),
            row["method"].casefold(),
            row["dataset"].casefold(),
        )
    )
    return {
        "population": population,
        "min_expected": min_expected,
        "tested_cells": len(rows),
        "fdr_method": "Benjamini-Hochberg",
        "lower_tail_test": "P(X <= observed_count | Poisson(expected_count))",
        "expected_count_formula": "N * p(task) * p(method) * p(dataset)",
        "marginal_denominator": (
            "all include_in_counts=true records, including records with other "
            "axes missing"
        ),
        "candidates_q_0_05": [row for row in rows if row["q_value"] <= 0.05],
        "candidates_q_0_10": [row for row in rows if row["q_value"] <= 0.10],
        "tested_cell_results": rows,
    }


def screen_pairwise_sparse_cells(
    tensor: CooccurrenceTensor, *, min_expected: float = 2.0
) -> dict[str, Any]:
    """Test pairwise cells and calculate observed-cell PMI as a secondary signal.

    Each pair uses only included papers where both fields passed specificity
    validation and have non-null values. The same pair-complete denominator is
    used for that pair's marginals, expectation, and PMI.
    """
    if not math.isfinite(min_expected) or min_expected < 0:
        raise ValueError("min_expected must be finite and nonnegative.")

    pair_results: dict[str, Any] = {}
    for left_axis, right_axis in PAIR_AXES:
        population = tensor.pair_populations[(left_axis, right_axis)]
        left_counts = {
            entity: count
            for (axis, entity, other_axis), count in tensor.pair_marginal_counts.items()
            if axis == left_axis and other_axis == right_axis
        }
        right_counts = {
            entity: count
            for (axis, entity, other_axis), count in tensor.pair_marginal_counts.items()
            if axis == right_axis and other_axis == left_axis
        }
        observed_counts = {
            (left, right): count
            for (axis, left, other_axis, right), count in tensor.pair_counts.items()
            if axis == left_axis and other_axis == right_axis
        }
        maximum_expected = (
            max(left_counts.values(), default=0)
            * max(right_counts.values(), default=0)
            / population
            if population > 0
            else 0.0
        )

        tested: list[dict[str, Any]] = []
        observed_cells: list[dict[str, Any]] = []
        sorted_right = sorted(
            (count, entity) for entity, count in right_counts.items()
        )
        sorted_right_counts = [count for count, _ in sorted_right]
        if population > 0:
            for left, left_count in left_counts.items():
                first_eligible = bisect.bisect_left(
                    sorted_right_counts,
                    min_expected * population / left_count,
                )
                for right_count, right in sorted_right[first_eligible:]:
                    mean = expected_pair_count(population, left_count, right_count)
                    observed = observed_counts.get((left, right), 0)
                    tested.append(
                        {
                            "left_entity": left,
                            "right_entity": right,
                            "observed_count": observed,
                            "expected_count": mean,
                            "p_value": poisson_lower_tail(observed, mean),
                        }
                    )

            for (left, right), observed in observed_counts.items():
                mean = expected_pair_count(
                    population, left_counts[left], right_counts[right]
                )
                pmi = math.log(observed / mean) if observed > 0 and mean > 0 else None
                observed_cells.append(
                    {
                        "left_entity": left,
                        "right_entity": right,
                        "observed_count": observed,
                        "expected_count": mean,
                        "pmi": pmi,
                    }
                )

        adjusted = benjamini_hochberg([row["p_value"] for row in tested])
        for row, q_value in zip(tested, adjusted, strict=True):
            row["q_value"] = q_value
        tested.sort(
            key=lambda row: (
                row["q_value"],
                row["p_value"],
                -row["expected_count"],
                row["left_entity"].casefold(),
                row["right_entity"].casefold(),
            )
        )
        observed_cells.sort(
            key=lambda row: (
                -(row["pmi"] if row["pmi"] is not None else -math.inf),
                -row["observed_count"],
                row["left_entity"].casefold(),
                row["right_entity"].casefold(),
            )
        )
        pair_key = f"{left_axis}_x_{right_axis}"
        pair_results[pair_key] = {
            "left_axis": left_axis,
            "right_axis": right_axis,
            "pair_complete_papers": population,
            "left_entity_count": len(left_counts),
            "right_entity_count": len(right_counts),
            "observed_non_empty_cells": len(observed_counts),
            "maximum_expected_count": maximum_expected,
            "tested_cells_expected_at_least_minimum": len(tested),
            "min_expected": min_expected,
            "fdr_method": "Benjamini-Hochberg",
            "candidates_q_0_05": [
                row for row in tested if row["q_value"] <= 0.05
            ],
            "candidates_q_0_10": [
                row for row in tested if row["q_value"] <= 0.10
            ],
            "tested_cell_results": tested,
            "observed_cell_pmi_results": observed_cells,
        }

    return {
        "method": "pairwise marginal-independence Poisson lower-tail tests",
        "expected_count_formula": "N_pair * p(left) * p(right)",
        "pmi_formula": "log(p(left,right) / (p(left) * p(right)))",
        "min_expected": min_expected,
        "pairwise_results": pair_results,
    }
