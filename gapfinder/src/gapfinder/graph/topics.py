from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

from gapfinder.extraction.normalize import cluster_entities
from gapfinder.extraction.schema import is_generic_claim_value
from gapfinder.graph.build import read_jsonl
from gapfinder.retrieval.dense import configured_model_name

TOPIC_FIELDS = ("task", "method", "dataset")
TOPIC_THRESHOLDS = (0.75, 0.80)
TOPIC_K_CANDIDATES = (30, 40, 50, 60, 75, 100)
SILHOUETTE_SAMPLE_SIZE = 1500
RANDOM_SEED = 20260930


def eligible_terms(extractions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Collect non-null, specificity-approved task/method/dataset surfaces."""
    terms: dict[str, set[str]] = {field: set() for field in TOPIC_FIELDS}
    for row in extractions:
        claims = row.get("claims")
        if not isinstance(claims, dict):
            raise ValueError("Extraction row has no claims object.")
        if not claims.get("include_in_counts", False):
            continue
        for field in TOPIC_FIELDS:
            evidence = claims.get(field)
            value = evidence.get("value") if isinstance(evidence, dict) else None
            if value and not is_generic_claim_value(field, value):
                terms[field].add(value)
    return {
        field: sorted(values, key=lambda value: (value.casefold(), value))
        for field, values in terms.items()
    }


def cluster_size_report(mapping: dict[str, str]) -> dict[str, Any]:
    """Summarize cluster-size distribution for one field's topic map."""
    sizes = Counter(mapping.values())
    counts = Counter(sizes.values())
    return {
        "surface_forms": len(mapping),
        "topic_count": len(sizes),
        "maximum_cluster_size": max(sizes.values(), default=0),
        "size_distribution": {
            str(size): counts[size] for size in sorted(counts)
        },
        "size_bins": {
            "1": sum(size == 1 for size in sizes.values()),
            "2-4": sum(2 <= size <= 4 for size in sizes.values()),
            "5-9": sum(5 <= size <= 9 for size in sizes.values()),
            "10-24": sum(10 <= size <= 24 for size in sizes.values()),
            "25-49": sum(25 <= size <= 49 for size in sizes.values()),
            "50-99": sum(50 <= size <= 99 for size in sizes.values()),
            "100+": sum(size >= 100 for size in sizes.values()),
        },
    }


def largest_clusters(
    mapping: dict[str, str], limit: int = 5
) -> list[dict[str, Any]]:
    """Return the largest clusters and their member surface forms."""
    members: dict[str, list[str]] = {}
    for surface, topic in mapping.items():
        members.setdefault(topic, []).append(surface)
    ordered = sorted(
        members.items(),
        key=lambda item: (-len(item[1]), item[0].casefold(), item[0]),
    )[:limit]
    return [
        {
            "topic": topic,
            "size": len(values),
            "members": sorted(values, key=lambda value: (value.casefold(), value)),
        }
        for topic, values in ordered
    ]


def kmeans_topic_map(
    terms: list[str],
    embeddings: np.ndarray,
    *,
    k_candidates: tuple[int, ...] = TOPIC_K_CANDIDATES,
    seed: int = RANDOM_SEED,
    silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Select k by sampled cosine silhouette, then label clusters by medoids."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(terms):
        raise ValueError("Embeddings must be a 2D matrix with one row per term.")
    if len(set(terms)) != len(terms):
        raise ValueError("Topic terms must be unique.")
    if len(terms) < 3:
        raise ValueError("At least three terms are required for topic clustering.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    candidates = sorted(
        {
            k
            for k in k_candidates
            if 2 <= k < len(terms)
        }
    )
    if not candidates:
        raise ValueError("No valid k candidates for the number of input terms.")

    evaluated: list[dict[str, Any]] = []
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    sample_size = min(silhouette_sample_size, len(terms))
    for k in candidates:
        estimator = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=5,
            max_iter=150,
            algorithm="lloyd",
        )
        labels = estimator.fit_predict(normalized)
        if len(set(int(label) for label in labels)) < 2:
            continue
        score = float(
            silhouette_score(
                normalized,
                labels,
                metric="cosine",
                sample_size=sample_size,
                random_state=seed,
            )
        )
        evaluated.append({"k": k, "cosine_silhouette": score})
        candidate = (score, -k, labels, estimator.cluster_centers_)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No valid KMeans clustering produced a silhouette score.")

    score, negative_k, labels, centers = best
    selected_k = -negative_k
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    mapping: dict[str, str] = {}
    for label, indices in groups.items():
        center = centers[label]
        medoid_index = min(
            indices,
            key=lambda index: (
                float(np.linalg.norm(normalized[index] - center)),
                terms[index].casefold(),
                terms[index],
            ),
        )
        topic_name = terms[medoid_index]
        mapping.update({terms[index]: topic_name for index in indices})

    return mapping, {
        "selection": "maximum sampled cosine silhouette; ties favor smaller k",
        "candidate_k": list(candidates),
        "evaluated": evaluated,
        "selected_k": selected_k,
        "selected_cosine_silhouette": score,
        "silhouette_sample_size": sample_size,
        "random_seed": seed,
    }


def build_topic_artifacts(
    extractions: list[dict[str, Any]],
    encoder: object,
    *,
    thresholds: tuple[float, ...] = TOPIC_THRESHOLDS,
    k_candidates: tuple[int, ...] = TOPIC_K_CANDIDATES,
    seed: int = RANDOM_SEED,
    silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build loose-threshold maps and silhouette-selected field topic maps."""
    terms_by_field = eligible_terms(extractions)
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise TypeError("Topic embedding encoder must provide encode().")
    vectors_by_field: dict[str, np.ndarray] = {}
    for field, terms in terms_by_field.items():
        if not terms:
            vectors_by_field[field] = np.empty((0, 0), dtype=np.float32)
        else:
            vectors_by_field[field] = np.asarray(
                encode(
                    terms,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

    threshold_reports: dict[str, Any] = {}
    for threshold in thresholds:
        maps: dict[str, dict[str, str]] = {}
        reports: dict[str, Any] = {}
        for field, terms in terms_by_field.items():
            mapping = cluster_entities(
                terms,
                vectors_by_field[field],
                similarity_threshold=threshold,
            )
            maps[field] = mapping
            reports[field] = {
                **cluster_size_report(mapping),
                "largest_clusters": largest_clusters(mapping),
            }
        threshold_reports[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "algorithm": "field-specific complete-linkage",
            "fields": reports,
        }

    topic_maps: dict[str, dict[str, str]] = {}
    topic_reports: dict[str, Any] = {}
    for field, terms in terms_by_field.items():
        mapping, selection = kmeans_topic_map(
            terms,
            vectors_by_field[field],
            k_candidates=k_candidates,
            seed=seed,
            silhouette_sample_size=silhouette_sample_size,
        )
        topic_maps[field] = mapping
        topic_reports[field] = {
            **selection,
            **cluster_size_report(mapping),
            "largest_clusters": largest_clusters(mapping),
        }

    topic_map_artifact = {
        "schema_version": 1,
        "kind": "topic-level sparsity map; does not replace fine-grained map",
        "embedding_model": configured_model_name(),
        "algorithm": "field-specific k-means; k selected by cosine silhouette",
        "random_seed": seed,
        "fields": topic_maps,
    }
    report = {
        "source_records": len(extractions),
        "fine_grained_included_papers": sum(
            1
            for row in extractions
            if (row.get("claims") or {}).get("include_in_counts") is True
        ),
        "eligible_terms_by_field": {
            field: len(terms) for field, terms in terms_by_field.items()
        },
        "loose_complete_linkage_thresholds": threshold_reports,
        "kmeans_topic_model": {
            "algorithm": "KMeans on normalized BGE embeddings",
            "selection": "maximum sampled cosine silhouette; ties favor smaller k",
            "field_reports": topic_reports,
        },
    }
    return topic_map_artifact, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build separate topic-level entity maps for sparsity analysis."
    )
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extractions = read_jsonl(args.extractions)

    from gapfinder.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(cache_dir=None)
    mapping, report = build_topic_artifacts(extractions, retriever.encoder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "topic_entity_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "topic_cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["kmeans_topic_model"]["field_reports"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
