from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

from gapfinder.config import DEDUPE_THRESHOLD


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    cleaned = title.strip().lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(a * a for a in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def dedupe_records(records: Iterable[dict[str, Any]], *, threshold: float = DEDUPE_THRESHOLD, return_stats: bool = False) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate records using DOI, normalized title, then embedding cosine similarity.

    Three stages are tracked explicitly so the final report can say how many papers were removed
    at each step. The embedding stage only applies when both records have an `embedding` list.
    """
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    kept: list[dict[str, Any]] = []
    stage_counts = {"doi_exact": 0, "title_normalized": 0, "embedding_cosine": 0}

    for record in records:
        doi = (record.get("doi") or "").strip().lower()
        title = normalize_title(record.get("title"))
        embedding = record.get("embedding")

        if doi:
            if doi in seen_dois:
                stage_counts["doi_exact"] += 1
                continue
            seen_dois.add(doi)

        if title:
            if title in seen_titles:
                stage_counts["title_normalized"] += 1
                continue
            seen_titles.add(title)

        if isinstance(embedding, list):
            for prior in kept:
                prior_embedding = prior.get("embedding")
                if isinstance(prior_embedding, list):
                    similarity = cosine_similarity(list(map(float, embedding)), list(map(float, prior_embedding)))
                    if similarity >= threshold:
                        stage_counts["embedding_cosine"] += 1
                        break
            else:
                kept.append(record)
                continue
            continue

        kept.append(record)

    if return_stats:
        return kept, stage_counts
    return kept


def make_record_key(record: dict[str, Any]) -> str:
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
