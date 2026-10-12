from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gapfinder.extraction.schema import is_generic_claim_value

AXES = ("task", "method", "dataset")
PAIR_AXES = (("task", "method"), ("task", "dataset"), ("method", "dataset"))


@dataclass(frozen=True)
class CooccurrenceTensor:
    """Sparse task-method-dataset counts from included extraction records."""

    included_records: int
    complete_records: int
    marginal_counts: dict[str, Counter[str]]
    cell_counts: Counter[tuple[str, str, str]]
    pair_counts: Counter[tuple[str, str, str, str]]
    pair_populations: Counter[tuple[str, str]]
    pair_marginal_counts: Counter[tuple[str, str, str]]
    citation_edges: tuple[tuple[str, str], ...]
    yearly_included_records: dict[int, int]
    specificity_rejections: Counter[str]

    @property
    def axis_entities(self) -> dict[str, tuple[str, ...]]:
        return {
            axis: tuple(sorted(self.marginal_counts[axis], key=lambda x: (x.casefold(), x)))
            for axis in AXES
        }

    @property
    def raw_cell_count(self) -> int:
        return (
            len(self.marginal_counts["task"])
            * len(self.marginal_counts["method"])
            * len(self.marginal_counts["dataset"])
        )

    @property
    def raw_sparsity(self) -> float:
        if self.raw_cell_count == 0:
            return 0.0
        return 1.0 - len(self.cell_counts) / self.raw_cell_count


def build_cooccurrence_tensor(
    extractions: list[dict[str, Any]],
    canonical_entities: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]] | None = None,
) -> CooccurrenceTensor:
    """Build sparse tensor, pairwise concept co-occurrence, and citation edges.

    Marginals count canonical values on included records even when another axis
    is missing. A tensor cell is counted only when all three axes are present.
    Repeated extraction rows and unknown canonical terms are rejected.
    """
    marginal_counts: dict[str, Counter[str]] = {
        axis: Counter() for axis in AXES
    }
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    pair_counts: Counter[tuple[str, str, str, str]] = Counter()
    pair_populations: Counter[tuple[str, str]] = Counter()
    pair_marginal_counts: Counter[tuple[str, str, str]] = Counter()
    specificity_rejections: Counter[str] = Counter()
    seen_papers: set[str] = set()
    citation_edges: set[tuple[str, str]] = set()
    yearly_included_records: Counter[int] = Counter()
    included_records = 0
    complete_records = 0

    for row_number, row in enumerate(extractions, start=1):
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError(f"Extraction row {row_number} has no paper_id.")
        if paper_id in seen_papers:
            raise ValueError(f"Duplicate extraction paper_id: {paper_id}")
        seen_papers.add(paper_id)

        claims = row.get("claims")
        if not isinstance(claims, dict) or not isinstance(
            claims.get("include_in_counts"), bool
        ):
            raise ValueError(f"Extraction row {row_number} has invalid claims.")
        if not claims["include_in_counts"]:
            continue

        included_records += 1
        corpus_record = None if corpus_by_id is None else corpus_by_id.get(paper_id)
        if corpus_by_id is not None and corpus_record is None:
            raise ValueError(f"Included extraction is absent from corpus: {paper_id}")
        if corpus_record is not None:
            year = corpus_record.get("year")
            if isinstance(year, int):
                yearly_included_records[year] += 1
            references = corpus_record.get("referenced_works") or []
            if not isinstance(references, list):
                raise ValueError(f"Corpus references must be a list for {paper_id}.")
            for reference in references:
                if isinstance(reference, str) and reference:
                    citation_edges.add((paper_id, reference))

        values: dict[str, str] = {}
        for axis in AXES:
            evidence = claims.get(axis)
            if evidence is None:
                continue
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"Extraction row {row_number} has invalid {axis} claim."
                )
            value = evidence.get("value")
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Extraction row {row_number} has an empty {axis} value."
                )
            if is_generic_claim_value(axis, value):
                specificity_rejections[axis] += 1
                continue
            field_mapping = _field_mapping(canonical_entities, axis)
            if value not in field_mapping:
                raise ValueError(
                    f"Canonical entity map is missing {axis} value {value!r}."
                )
            canonical = field_mapping[value]
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError(
                    f"Canonical entity map has an invalid value for {value!r}."
                )
            values[axis] = canonical
            marginal_counts[axis][canonical] += 1

        for left_axis, right_axis in PAIR_AXES:
            left, right = values.get(left_axis), values.get(right_axis)
            if left is not None and right is not None:
                pair_populations[(left_axis, right_axis)] += 1
                pair_marginal_counts[(left_axis, left, right_axis)] += 1
                pair_marginal_counts[(right_axis, right, left_axis)] += 1
                pair_counts[(left_axis, left, right_axis, right)] += 1
        if all(axis in values for axis in AXES):
            cell = (values["task"], values["method"], values["dataset"])
            cell_counts[cell] += 1
            complete_records += 1

    return CooccurrenceTensor(
        included_records=included_records,
        complete_records=complete_records,
        marginal_counts=marginal_counts,
        cell_counts=cell_counts,
        pair_counts=pair_counts,
        pair_populations=pair_populations,
        pair_marginal_counts=pair_marginal_counts,
        citation_edges=tuple(sorted(citation_edges)),
        yearly_included_records=dict(sorted(yearly_included_records.items())),
        specificity_rejections=specificity_rejections,
    )


def _field_mapping(
    canonical_entities: dict[str, Any], field_name: str
) -> dict[str, str]:
    """Read the versioned field map, while accepting legacy flat maps in tests."""
    fields = canonical_entities.get("fields")
    if fields is None:
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in canonical_entities.items()
        ):
            raise ValueError("Legacy canonical entity map must contain string pairs.")
        return {
            key: value
            for key, value in canonical_entities.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    mapping = fields.get(field_name)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError(f"Canonical entity map has an invalid {field_name} mapping.")
    return mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a UTF-8 JSONL file, with line-specific errors."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def tensor_as_json(
    tensor: CooccurrenceTensor, canonical_entities: dict[str, Any]
) -> dict[str, Any]:
    """Serialize tensor and graph counts as an auditable sparse artifact."""
    dimensions = {axis: len(tensor.marginal_counts[axis]) for axis in AXES}
    fields = canonical_entities.get("fields")
    if isinstance(fields, dict):
        surface_count = sum(
            len(value) for value in fields.values() if isinstance(value, dict)
        )
        target_count = sum(
            len(set(value.values()))
            for value in fields.values()
            if isinstance(value, dict)
        )
    else:
        surface_count = len(canonical_entities)
        target_count = len(set(canonical_entities.values()))
    pair_edges = [
        {
            "source_axis": left_axis,
            "source_entity": left,
            "target_axis": right_axis,
            "target_entity": right,
            "weight": count,
        }
        for (left_axis, left, right_axis, right), count in sorted(
            tensor.pair_counts.items()
        )
    ]
    return {
        "summary": {
            "included_records": tensor.included_records,
            "complete_records": tensor.complete_records,
            "non_empty_tensor_cells": len(tensor.cell_counts),
            "axis_entity_counts": dimensions,
            "raw_tensor_cells": tensor.raw_cell_count,
            "raw_tensor_sparsity": tensor.raw_sparsity,
            "canonical_map_surface_forms": surface_count,
            "canonical_map_distinct_targets_within_field": target_count,
            "posthoc_specificity_rejections": dict(
                sorted(tensor.specificity_rejections.items())
            ),
            "pairwise_concept_edges": len(pair_edges),
            "paper_citation_edges": len(tensor.citation_edges),
            "yearly_included_records": {
                str(year): count
                for year, count in tensor.yearly_included_records.items()
            },
        },
        "axis_entities": {
            axis: [
                {"entity": entity, "paper_count": tensor.marginal_counts[axis][entity]}
                for entity in tensor.axis_entities[axis]
            ]
            for axis in AXES
        },
        "non_empty_cells": [
            {
                "task": task,
                "method": method,
                "dataset": dataset,
                "observed_count": count,
            }
            for (task, method, dataset), count in sorted(tensor.cell_counts.items())
        ],
        "pairwise_concept_edges": pair_edges,
        "paper_citation_edges": [
            {"source_paper": source, "target_paper": target}
            for source, target in tensor.citation_edges
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v4 sparse concept tensor and lower-tail sparsity results."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--canonical-entities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-expected", type=float, default=2.0)
    args = parser.parse_args()

    extractions = read_jsonl(args.extractions)
    corpus_rows = read_jsonl(args.corpus)
    corpus_by_id = {
        row["id"]: row
        for row in corpus_rows
        if isinstance(row.get("id"), str)
    }
    if len(corpus_by_id) != len(corpus_rows):
        raise ValueError("Corpus has missing or duplicate paper IDs.")
    canonical_entities = json.loads(
        args.canonical_entities.read_text(encoding="utf-8")
    )
    if not isinstance(canonical_entities, dict):
        raise ValueError("Canonical entity artifact must be a JSON object.")

    tensor = build_cooccurrence_tensor(
        extractions, canonical_entities, corpus_by_id
    )
    tensor_payload = tensor_as_json(tensor, canonical_entities)
    from gapfinder.graph.sparsity import (
        screen_pairwise_sparse_cells,
        screen_sparse_cells,
    )

    sparsity = screen_sparse_cells(
        tensor, min_expected=args.min_expected
    )
    pairwise_sparsity = screen_pairwise_sparse_cells(
        tensor, min_expected=args.min_expected
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "concept_tensor.json", tensor_payload)
    _write_json(args.output_dir / "sparsity_results.json", sparsity)
    _write_json(args.output_dir / "pairwise_sparsity_results.json", pairwise_sparsity)
    print(json.dumps({
        **tensor_payload["summary"],
        "tested_cells": sparsity["tested_cells"],
        "candidates_q_0_05": len(sparsity["candidates_q_0_05"]),
        "candidates_q_0_10": len(sparsity["candidates_q_0_10"]),
        "pairwise_summary": {
            pair: {
                "pair_complete_papers": result["pair_complete_papers"],
                "observed_non_empty_cells": result["observed_non_empty_cells"],
                "maximum_expected_count": result["maximum_expected_count"],
                "tested_cells": result["tested_cells_expected_at_least_minimum"],
                "candidates_q_0_05": len(result["candidates_q_0_05"]),
                "candidates_q_0_10": len(result["candidates_q_0_10"]),
            }
            for pair, result in pairwise_sparsity["pairwise_results"].items()
        },
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
