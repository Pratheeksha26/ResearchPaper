from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

Item = TypeVar("Item")


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Item]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[Item, float]]:
    """Fuse ranked item lists with RRF; ranks are one-based.

    Optional weights support explicitly labeled fusion ablations; the default
    unweighted form remains the primary RRF method.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative.")
    if weights is not None and any(
        weight < 0 for weight in weights.values()
    ):
        raise ValueError("RRF weights must be non-negative.")
    scores: dict[Item, float] = {}
    first_seen: dict[Item, tuple[int, int]] = {}
    for rank_group, (name, items) in enumerate(rankings.items()):
        seen: set[Item] = set()
        rank = 0
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            weight = 1.0 if weights is None else weights.get(name, 1.0)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            first_seen.setdefault(item, (rank_group, rank))
    return sorted(
        scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]])
    )
