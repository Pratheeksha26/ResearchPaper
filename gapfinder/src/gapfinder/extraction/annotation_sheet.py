from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from gapfinder.extraction.schema import ClaimExtraction

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = PACKAGE_ROOT / "corpus.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "experiments" / "annotation" / "extraction_sample.csv"
ANNOTATION_FIELDS = tuple(f"{field}_label" for field in ClaimExtraction.model_fields)


def create_annotation_sheet(
    corpus_path: Path = DEFAULT_CORPUS,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> int:
    """Write a deterministic year-stratified sample with blank label cells."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    papers: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                paper = json.loads(line)
                if str(paper.get("abstract") or "").strip():
                    papers.append(paper)
    by_year: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        year = str(paper.get("year") or "unknown")
        by_year.setdefault(year, []).append(paper)
    target_size = min(sample_size, len(papers))
    preferred_minimums = {
        year: min(5, len(year_papers))
        for year, year_papers in by_year.items()
        if year.isdigit() and 2019 <= int(year) <= 2025
    }
    if target_size >= sum(preferred_minimums.values()):
        allocations = {year: 0 for year in by_year}
        allocations.update(preferred_minimums)
        for year, year_papers in by_year.items():
            if allocations[year] == 0 and sum(allocations.values()) < target_size:
                allocations[year] = min(1, len(year_papers))
    else:
        minimum_per_year = 1 if target_size >= len(by_year) else 0
        allocations = {
            year: min(minimum_per_year, len(year_papers))
            for year, year_papers in by_year.items()
        }
    remainder = target_size - sum(allocations.values())
    if remainder < 0:
        raise ValueError(
            "sample_size is too small to guarantee the required per-year minimums."
        )
    remaining_capacity = sum(
        len(by_year[year]) - allocations[year] for year in by_year
    )
    exact_additions = {
        year: (
            remainder * (len(year_papers) - allocations[year]) / remaining_capacity
            if remaining_capacity
            else 0.0
        )
        for year, year_papers in by_year.items()
    }
    base_additions = {
        year: min(
            len(by_year[year]) - allocations[year],
            math.floor(exact_additions[year]),
        )
        for year in by_year
    }
    for year, amount in base_additions.items():
        allocations[year] += amount
    remainder -= sum(base_additions.values())
    allocation_order = sorted(
        by_year,
        key=lambda year: (
            -(exact_additions[year] - math.floor(exact_additions[year])),
            year,
        ),
    )
    for year in allocation_order:
        if remainder == 0:
            break
        if allocations[year] < len(by_year[year]):
            allocations[year] += 1
            remainder -= 1
    rng = random.Random(seed)
    sample = [
        paper
        for year in sorted(by_year)
        for paper in rng.sample(by_year[year], allocations[year])
    ]
    rng.shuffle(sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["paper_id", "title", "year", "abstract", *ANNOTATION_FIELDS]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for paper in sample:
            writer.writerow(
                {
                    "paper_id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "abstract": paper.get("abstract", ""),
                    **{field: "" for field in ANNOTATION_FIELDS},
                }
            )
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = create_annotation_sheet(
        args.corpus,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Wrote {count} rows with blank labels to {args.output}")


if __name__ == "__main__":
    main()
