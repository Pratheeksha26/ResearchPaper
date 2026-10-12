from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from gapfinder.extraction.schema import ClaimExtraction, is_generic_claim_value

ENTITY_FIELDS = ("task", "method", "dataset", "metric", "setting")
DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.99


def cluster_entities(
    terms: Sequence[str],
    embeddings: np.ndarray,
    *,
    similarity_threshold: float = DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Cluster terms with deterministic complete-linkage constraints.

    Complete linkage merges clusters only when every cross-cluster cosine
    similarity meets the threshold, preventing weak intermediate links from
    joining unrelated endpoints.
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(terms):
        raise ValueError("Embeddings must be a 2D matrix with one row per term.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("Similarity threshold must be between zero and one.")
    if len(set(terms)) != len(terms):
        raise ValueError("Entity terms must be unique.")
    if not terms:
        return {}

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    term_order = sorted(
        range(len(terms)), key=lambda index: (terms[index].casefold(), terms[index])
    )
    ordered_terms = [terms[index] for index in term_order]
    ordered_vectors = normalized[term_order]
    if len(ordered_terms) == 1:
        return {ordered_terms[0]: ordered_terms[0]}

    hierarchy = linkage(pdist(ordered_vectors, metric="cosine"), method="complete")
    cluster_ids = fcluster(
        hierarchy,
        t=1.0 - similarity_threshold,
        criterion="distance",
    )
    clusters: dict[int, list[str]] = {}
    for term, cluster_id in zip(ordered_terms, cluster_ids, strict=True):
        clusters.setdefault(int(cluster_id), []).append(term)

    mapping: dict[str, str] = {}
    for members in clusters.values():
        canonical = min(
            members,
            key=lambda term: (term.casefold(), len(term), term),
        )
        mapping.update({term: canonical for term in members})
    return mapping


def normalize_claim_entities(
    claims: Sequence[ClaimExtraction],
    encoder: object,
    *,
    similarity_threshold: float = DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
) -> dict[str, dict[str, str]]:
    """Normalize each claim field independently using complete-linkage clusters."""
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise TypeError("Entity normalization encoder must provide encode().")

    field_mappings: dict[str, dict[str, str]] = {}
    for field_name in ENTITY_FIELDS:
        terms = sorted(
            {
                value
                for extraction in claims
                if (value := getattr(extraction, field_name).value)
                and not is_generic_claim_value(field_name, value)
            },
            key=lambda value: (value.casefold(), value),
        )
        if not terms:
            field_mappings[field_name] = {}
            continue
        vectors = np.asarray(
            encode(
                terms,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        field_mappings[field_name] = cluster_entities(
            terms,
            vectors,
            similarity_threshold=similarity_threshold,
        )
    return field_mappings
