from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def yearly_concept_share(
    records: Iterable[dict[str, Any]],
    concept: str,
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict[int, dict[str, float | int]]:
    """Estimate annual paper-level concept prevalence with percentile bootstrap CIs.

    The denominator is papers in each year, not the raw annual record count as a
    proxy for literature volume. This is required for a year-stratified corpus.
    """
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    concept_key = concept.strip().casefold()
    if not concept_key:
        raise ValueError("concept must be nonempty.")
    grouped: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        year = int(record.get("year") or 0)
        if not year:
            continue
        values = record.get("concepts") or []
        present = any(
            str(value).strip().casefold() == concept_key for value in values
        )
        grouped[year].append(present)

    rng = random.Random(seed)
    output: dict[int, dict[str, float | int]] = {}
    for year, observations in sorted(grouped.items()):
        n = len(observations)
        samples = [
            sum(observations[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(bootstrap_resamples)
        ]
        samples.sort()
        output[year] = {
            "papers": n,
            "concept_papers": sum(observations),
            "share": sum(observations) / n,
            "ci95_low": samples[int(0.025 * (bootstrap_resamples - 1))],
            "ci95_high": samples[int(0.975 * (bootstrap_resamples - 1))],
        }
    return output
