from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests

from gapfinder import config
from gapfinder.retrieval.bm25 import BM25Index, document_text
from gapfinder.retrieval.dense import (
    DenseRetriever,
    configured_model_name,
)
from gapfinder.retrieval.fusion import reciprocal_rank_fusion

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = PACKAGE_ROOT / "corpus.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "experiments" / "runs" / "retrieval"
SURVEY_TERMS = ("survey", "review", "systematic mapping", "systematic literature")
DEFAULT_RECALL_CUTOFFS = (10, 20, 50)
DEFAULT_NDCG_CUTOFFS = (10, 20)
MIN_RELEVANT_DOCUMENTS = 3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928
RRF_K = 60
WEIGHTED_RRF_DENSE_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
SECONDARY_RECHECK_LIMIT = 25


def _canonical_id(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    return text.rsplit("/", 1)[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path.name}:{line_number}.") from exc
                if isinstance(record, dict):
                    records.append(record)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_surveys(
    records: Sequence[dict[str, Any]], *, max_surveys: int = 20
) -> list[dict[str, Any]]:
    """Select actual survey/review works from a corpus using title and reference fields."""
    candidates = [
        record
        for record in records
        if record.get("title")
        and record.get("referenced_works")
        and not record.get("synthetic")
        and any(term in str(record["title"]).lower() for term in SURVEY_TERMS)
        and re.fullmatch(r"W\d+", _canonical_id(record.get("id"))) is not None
    ]
    candidates.sort(
        key=lambda record: (
            -(int(record.get("year") or 0)),
            _canonical_id(record.get("id")),
        )
    )
    return candidates[:max_surveys]


def select_expanded_survey_candidates(
    records: Sequence[dict[str, Any]], *, max_surveys: int = 20
) -> list[dict[str, Any]]:
    """Select GNN survey sources from Phase-1 records and resolved references."""
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        work_id = _canonical_id(record.get("id"))
        title = str(record.get("title") or "").lower()
        if (
            not record.get("title")
            or not record.get("referenced_works")
            or record.get("synthetic")
            or re.fullmatch(r"W\d+", work_id) is None
            or not any(term in title for term in SURVEY_TERMS)
            or not any(
                term in title
                for term in ("graph neural", "graph convolutional", "graph learning")
            )
        ):
            continue
        candidates_by_id[work_id] = record
    candidates = list(candidates_by_id.values())
    candidates.sort(
        key=lambda record: (
            -(int(record.get("year") or 0)),
            _canonical_id(record.get("id")),
        )
    )
    return candidates[:max_surveys]


class OpenAlexSingletons:
    """Polite singleton work lookups backed by a local cache without storing credentials."""

    def __init__(self, cache_file: Path, *, minimum_interval: float = 0.15) -> None:
        self.cache_file = cache_file
        self.minimum_interval = minimum_interval
        self.cache: dict[str, dict[str, Any]] = {}
        self.not_found: set[str] = set()
        self._last_request = 0.0
        if cache_file.is_file():
            try:
                loaded = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("works"), dict):
                        self.cache = {
                            str(key): value
                            for key, value in loaded["works"].items()
                            if isinstance(value, dict)
                        }
                        self.not_found = set(loaded.get("not_found") or [])
                    else:
                        self.cache = {
                            str(key): value
                            for key, value in loaded.items()
                            if isinstance(value, dict)
                        }
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def get(self, work_id: str) -> dict[str, Any] | None:
        """Fetch one OpenAlex work, reusing cached responses and retrying transient errors."""
        canonical = _canonical_id(work_id)
        if not re.fullmatch(r"W\d+", canonical):
            raise ValueError("OpenAlex singleton lookup requires a work ID.")
        if canonical in self.cache:
            return self.cache[canonical]
        if canonical in self.not_found:
            return None

        base_url = config.OPENALEX_BASE_URL.rstrip("/")
        url = f"{base_url}/works/{quote(canonical)}"
        params = {"api_key": config.OPENALEX_API_KEY} if config.OPENALEX_API_KEY else None
        headers = {"User-Agent": f"gapfinder/0.1 ({config.EMAIL})"}
        for attempt in range(6):
            pause = self.minimum_interval - (time.monotonic() - self._last_request)
            if pause > 0:
                time.sleep(pause)
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                self._last_request = time.monotonic()
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 5:
                        raise RuntimeError(
                            f"OpenAlex singleton lookup exhausted retries (HTTP {response.status_code})."
                        )
                    retry_after = response.headers.get("Retry-After")
                    delay = min(30.0, float(retry_after)) if retry_after else 2**attempt
                    time.sleep(delay)
                    continue
                if response.status_code == 404:
                    self.not_found.add(canonical)
                    return None
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"OpenAlex singleton lookup returned HTTP {response.status_code}."
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or _canonical_id(payload.get("id")) != canonical:
                    raise RuntimeError(
                        f"OpenAlex returned mismatched metadata for {canonical}."
                    )
                self.cache[canonical] = payload
                return payload
            except (requests.RequestException, ValueError) as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"OpenAlex singleton lookup failed for {canonical} "
                        f"({type(exc).__name__})."
                    ) from None
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAlex singleton lookup failed for {canonical}.")

    def get_many(self, work_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Fetch works in documented OpenAlex OR-filter batches of at most 100 IDs."""
        canonical_ids = sorted({_canonical_id(work_id) for work_id in work_ids})
        if any(not re.fullmatch(r"W\d+", work_id) for work_id in canonical_ids):
            raise ValueError("OpenAlex batch lookup requires work IDs.")
        missing = [
            work_id
            for work_id in canonical_ids
            if work_id not in self.cache
        ]
        base_url = config.OPENALEX_BASE_URL.rstrip("/") + "/works"
        headers = {"User-Agent": f"gapfinder/0.1 ({config.EMAIL})"}
        select = ",".join(
            (
                "id",
                "title",
                "doi",
                "publication_year",
                "primary_location",
                "referenced_works",
                "cited_by_count",
                "ids",
                "abstract_inverted_index",
            )
        )
        for start in range(0, len(missing), 100):
            batch = missing[start : start + 100]
            params: dict[str, Any] = {
                "filter": f"ids.openalex:{'|'.join(batch)}",
                "per_page": 100,
                "select": select,
            }
            if config.OPENALEX_API_KEY:
                params["api_key"] = config.OPENALEX_API_KEY
            for attempt in range(6):
                pause = self.minimum_interval - (time.monotonic() - self._last_request)
                if pause > 0:
                    time.sleep(pause)
                try:
                    response = requests.get(
                        base_url, params=params, headers=headers, timeout=60
                    )
                    self._last_request = time.monotonic()
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt == 5:
                            raise RuntimeError(
                                "OpenAlex batch lookup exhausted transient retries."
                            )
                        retry_after = response.headers.get("Retry-After")
                        delay = min(30.0, float(retry_after)) if retry_after else 2**attempt
                        time.sleep(delay)
                        continue
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"OpenAlex batch lookup returned HTTP {response.status_code}."
                        )
                    payload = response.json()
                    results = payload.get("results", []) if isinstance(payload, dict) else []
                    if not isinstance(results, list) or len(results) > 100:
                        raise RuntimeError("OpenAlex batch lookup returned invalid results.")
                    returned: set[str] = set()
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        work_id = _canonical_id(item.get("id"))
                        if work_id not in batch:
                            raise RuntimeError(
                                "OpenAlex batch lookup returned an unexpected work."
                            )
                        self.cache[work_id] = item
                        returned.add(work_id)
                    self.not_found.difference_update(set(batch) - returned)
                    break
                except (requests.RequestException, ValueError) as exc:
                    if attempt == 5:
                        raise RuntimeError(
                            "OpenAlex batch lookup failed "
                            f"({type(exc).__name__})."
                        ) from None
                    time.sleep(2**attempt)
        return {
            work_id: self.cache[work_id]
            for work_id in canonical_ids
            if work_id in self.cache
        }

    def save(self) -> None:
        """Persist successful responses with ID-only keys, never request URLs or keys."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps(
                {"works": self.cache, "not_found": sorted(self.not_found)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _record_from_openalex(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only bibliographic fields needed by the saved evaluation source metadata."""
    location = payload.get("primary_location") or {}
    source = location.get("source") or {}
    ids = payload.get("ids") or {}
    from gapfinder.data.openalex import reconstruct_abstract

    return {
        "id": payload.get("id"),
        "title": payload.get("title"),
        "doi": payload.get("doi"),
        "year": payload.get("publication_year"),
        "venue": source.get("display_name"),
        "referenced_works": payload.get("referenced_works") or [],
        "cited_by_count": payload.get("cited_by_count", 0),
        "abstract": reconstruct_abstract(payload),
        "url": ids.get("openalex"),
    }


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Compute binary Recall@k, returning None when a query has no relevant candidates."""
    if not relevant:
        return None
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Compute binary Precision@k, returning None when a query has no relevant candidates."""
    if not relevant:
        return None
    return len(set(ranking[:k]) & relevant) / k


def mean_average_precision(ranking: Sequence[str], relevant: set[str]) -> float | None:
    """Compute average precision over the full ranked pool."""
    if not relevant:
        return None
    hits = 0
    precision_sum = 0.0
    for rank, work_id in enumerate(ranking, start=1):
        if work_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Compute binary nDCG@k for a ranked list."""
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, work_id in enumerate(ranking[:k])
        if work_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in their original order."""
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("P-values must be between zero and one.")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running_max = 0.0
    count = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * p_value))
        adjusted[original_index] = running_max
    return adjusted


def _lookup_many(
    lookup: OpenAlexSingletons, work_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    batch_lookup = getattr(lookup, "get_many", None)
    if callable(batch_lookup):
        return batch_lookup(work_ids)
    resolved: dict[str, dict[str, Any]] = {}
    for work_id in work_ids:
        payload = lookup.get(work_id)
        if payload is not None:
            resolved[_canonical_id(work_id)] = payload
    return resolved


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array, size=(BOOTSTRAP_RESAMPLES, len(array)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return float(array.mean()), float(lower), float(upper)


def _paired_wilcoxon_pvalue(left: Sequence[float], right: Sequence[float]) -> float:
    """Return an exact two-sided paired Wilcoxon signed-rank sign-flip p-value."""
    if len(left) != len(right):
        raise ValueError("Paired Wilcoxon inputs must have equal lengths.")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    nonzero = [(abs(value), value) for value in differences if value != 0.0]
    if not nonzero:
        return 1.0
    nonzero.sort(key=lambda pair: pair[0])
    doubled_ranks: list[int] = []
    index = 0
    while index < len(nonzero):
        end = index + 1
        while end < len(nonzero) and nonzero[end][0] == nonzero[index][0]:
            end += 1
        doubled_rank = (index + 1) + end
        doubled_ranks.extend([doubled_rank] * (end - index))
        index = end

    observed = sum(
        rank if value > 0 else -rank
        for rank, (_absolute, value) in zip(doubled_ranks, nonzero)
    )
    distribution: Counter[int] = Counter({0: 1})
    for rank in doubled_ranks:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + rank] += count
            updated[total - rank] += count
        distribution = updated
    extreme = sum(
        count for total, count in distribution.items() if abs(total) >= abs(observed)
    )
    return min(1.0, extreme / (2 ** len(doubled_ranks)))


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metadata_record(record: dict[str, Any], *, origin: str, status: str) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "doi": record.get("doi"),
        "year": record.get("year"),
        "url": record.get("url"),
        "referenced_works_count": len(record.get("referenced_works") or []),
        "metadata_source": origin,
        "lookup_status": status,
    }


def run_evaluation(
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    output_dir: Path = DEFAULT_OUTPUT,
    max_surveys: int = 20,
    min_relevant_documents: int = MIN_RELEVANT_DOCUMENTS,
    cutoffs: Sequence[int] = DEFAULT_RECALL_CUTOFFS,
    model_name: str | None = None,
    exclude_unresolved_fraction_above: float | None = None,
) -> dict[str, Any]:
    """Run a citation-grounded survey recovery experiment and persist all artifacts."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("At least one positive evaluation cutoff is required.")
    if min_relevant_documents <= 0:
        raise ValueError("The minimum relevant-document threshold must be positive.")
    if (
        exclude_unresolved_fraction_above is not None
        and not 0.0 <= exclude_unresolved_fraction_above <= 1.0
    ):
        raise ValueError("Unresolved-reference fraction threshold must be in [0, 1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(corpus_path)
    by_id = {
        _canonical_id(record.get("id")): record
        for record in records
        if re.fullmatch(r"W\d+", _canonical_id(record.get("id")))
        and not record.get("synthetic")
    }
    seed_surveys = select_surveys(records, max_surveys=max_surveys)
    lookup = OpenAlexSingletons(config.CACHE_DIR / "retrieval" / "openalex_singletons.json")
    model = model_name or configured_model_name()
    source_metadata: list[dict[str, Any]] = []
    resolved_references: dict[str, dict[str, Any]] = {}
    ground_truth: list[dict[str, Any]] = []
    survey_source_records: dict[str, dict[str, Any]] = {}
    all_reference_ids: set[str] = set()
    survey_candidates_available = 0
    screened_survey_count = 0
    unresolved_relevant_ids: set[str] = set()
    lookup_error: str | None = None

    try:
        live_surveys = _lookup_many(
            lookup, sorted(_canonical_id(item["id"]) for item in seed_surveys)
        )
        seed_reference_ids: set[str] = set()
        live_survey_records: dict[str, dict[str, Any]] = {}
        for survey in seed_surveys:
            sid = _canonical_id(survey["id"])
            live_payload = live_surveys.get(sid)
            if live_payload is None:
                raise RuntimeError(f"OpenAlex survey work {sid} could not be resolved.")
            live_survey = _record_from_openalex(live_payload)
            live_survey_records[sid] = live_survey
            seed_reference_ids.update(
                _canonical_id(item)
                for item in live_survey["referenced_works"]
                if _canonical_id(item)
            )

        seed_reference_payloads = _lookup_many(lookup, sorted(seed_reference_ids))
        seed_resolved_references = {
            work_id: _record_from_openalex(payload)
            for work_id, payload in seed_reference_payloads.items()
        }
        source_records = dict(by_id)
        source_records.update(seed_resolved_references)
        source_records.update(live_survey_records)
        available_candidates = select_expanded_survey_candidates(
            list(source_records.values()),
            max_surveys=max(len(source_records), max_surveys),
        )
        survey_candidates_available = len(available_candidates)
        selected_surveys = available_candidates[:max_surveys]
        screened_survey_count = len(selected_surveys)
        selected_records = {
            _canonical_id(record.get("id")): record for record in selected_surveys
        }
        selected_records.update(
            {
                sid: live_survey_records[sid]
                for sid in selected_records
                if sid in live_survey_records
            }
        )
        survey_source_records = selected_records
        qualified_source_ids = {
            sid
            for sid, record in selected_records.items()
            if str(record.get("abstract") or "").strip()
        }
        query_reference_ids: set[str] = set()
        for sid in qualified_source_ids:
            query_reference_ids.update(
                _canonical_id(item)
                for item in selected_records[sid].get("referenced_works", [])
                if _canonical_id(item)
            )
        reference_payloads = _lookup_many(lookup, sorted(query_reference_ids))
        all_reference_ids = query_reference_ids
        first_route_unresolved = set(query_reference_ids) - set(reference_payloads)
        secondary_recheck_rows: list[dict[str, Any]] = []
        for work_id in sorted(first_route_unresolved)[:SECONDARY_RECHECK_LIMIT]:
            payload = lookup.get(work_id)
            if payload is None:
                secondary_recheck_rows.append(
                    {
                        "work_id": work_id,
                        "status": (
                            "singleton returned HTTP 404"
                            if work_id in getattr(lookup, "not_found", set())
                            else "still unresolved by singleton endpoint"
                        ),
                    }
                )
            else:
                reference_payloads[work_id] = payload
                secondary_recheck_rows.append(
                    {
                        "work_id": work_id,
                        "status": (
                            "resolved by singleton endpoint"
                            if _canonical_id(payload.get("id")) == work_id
                            else "singleton returned a different canonical ID"
                        ),
                        "canonical_work_id": _canonical_id(payload.get("id")),
                    }
                )
        unresolved_relevant_ids.update(set(query_reference_ids) - set(reference_payloads))
        resolved_references = {
            work_id: _record_from_openalex(payload)
            for work_id, payload in reference_payloads.items()
        }
        for sid, source in selected_records.items():
            source_references = sorted(
                {
                    _canonical_id(item)
                    for item in source.get("referenced_works", [])
                    if _canonical_id(item)
                }
            )
            ground_truth.append(
                {
                    "survey_id": sid,
                    "survey_title": source.get("title"),
                    "query": source.get("title"),
                    "survey_year": source.get("year"),
                    "survey_doi": source.get("doi"),
                    "source": "OpenAlex work.referenced_works",
                    "referenced_work_ids": source_references,
                    "referenced_count": len(source_references),
                    "source_abstract_available": bool(
                        str(source.get("abstract") or "").strip()
                    ),
                }
            )
            source_metadata.append(
                _metadata_record(
                    source,
                    origin=(
                        "Phase-1 OpenAlex corpus"
                        if sid in by_id
                        else "OpenAlex OR-filter list API"
                    ),
                    status="screened survey source",
                )
            )
        for item in ground_truth:
            item["resolved_reference_ids"] = sorted(
                set(item["referenced_work_ids"]) & set(resolved_references)
            )
            item["unresolved_reference_ids"] = (
                sorted(set(item["referenced_work_ids"]) - set(resolved_references))
                if item["source_abstract_available"]
                else []
            )
    except Exception as exc:
        lookup_error = f"{type(exc).__name__}: OpenAlex source/reference resolution failed."
    finally:
        lookup.save()
        _write_jsonl(output_dir / "source_metadata.jsonl", source_metadata)
        _write_jsonl(output_dir / "resolved_references.jsonl", resolved_references.values())
        _write_jsonl(output_dir / "ground_truth.jsonl", ground_truth)

    if lookup_error:
        status = {
            "status": "blocked",
            "blocker": lookup_error,
            "metrics": "OpenAlex source/reference lookups did not complete.",
            "source_survey_count": len(source_metadata),
            "unique_cited_work_count": len(all_reference_ids),
            "resolved_cited_work_count": len(resolved_references),
        }
        _write_csv(
            output_dir / "metrics.csv",
            ["method", "metric", "k", "mean", "ci_lower", "ci_upper", "n_queries", "status"],
            [],
        )
        (output_dir / "run.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return status

    pool_by_id = {
        work_id: record
        for work_id, record in by_id.items()
        if str(record.get("abstract") or "").strip()
    }
    for work_id, record in survey_source_records.items():
        if str(record.get("abstract") or "").strip():
            pool_by_id[work_id] = record
    no_abstract_phase1_ids = sorted(set(by_id) - set(pool_by_id))
    no_abstract_reference_ids = sorted(
        work_id
        for work_id, record in resolved_references.items()
        if not str(record.get("abstract") or "").strip()
    )
    for work_id, record in resolved_references.items():
        if str(record.get("abstract") or "").strip():
            pool_by_id[work_id] = record
    candidates = [pool_by_id[work_id] for work_id in sorted(pool_by_id)]
    candidate_ids = [_canonical_id(record["id"]) for record in candidates]
    candidate_id_set = set(candidate_ids)
    for item in ground_truth:
        relevant_ids = sorted(
            (set(item["referenced_work_ids"]) & candidate_id_set)
            - {item["survey_id"]}
        )
        item["relevant_candidate_ids"] = relevant_ids
        item["relevant_candidate_count"] = len(relevant_ids)
        item["self_reference_excluded"] = item["survey_id"] in set(
            item["referenced_work_ids"]
        )
        item["no_abstract_reference_ids"] = sorted(
            set(item["referenced_work_ids"]) & set(no_abstract_reference_ids)
        )
        item["unresolved_relevant_ids"] = sorted(
            set(item["referenced_work_ids"]) & unresolved_relevant_ids
        )

    qualification_rows: list[dict[str, Any]] = []
    for item in ground_truth:
        relevant_count = int(item["relevant_candidate_count"])
        has_abstract = bool(item["source_abstract_available"])
        qualified = has_abstract and relevant_count >= min_relevant_documents
        if not has_abstract:
            reason = "excluded: source survey has no abstract in OpenAlex"
        elif not qualified:
            reason = (
                "excluded: below minimum relevant-document threshold "
                f"({relevant_count} < {min_relevant_documents})"
            )
        else:
            reason = "qualified: abstract present and relevant-document threshold met"
        item["qualified"] = qualified
        item["qualification_reason"] = reason
        qualification_rows.append(
            {
                "survey_id": item["survey_id"],
                "survey_title": item["survey_title"],
                "referenced_count": item["referenced_count"],
                "abstract_relevant_count": relevant_count,
                "minimum_relevant_documents": min_relevant_documents,
                "source_abstract_available": has_abstract,
                "qualified": qualified,
                "reason": reason,
            }
        )
    qualified_before_sensitivity = [
        item for item in ground_truth if item["qualified"]
    ]
    excluded_by_unresolved_sensitivity = 0
    if exclude_unresolved_fraction_above is not None:
        qualified_ground_truth = []
        for item in qualified_before_sensitivity:
            unresolved_fraction = (
                len(item["unresolved_reference_ids"]) / item["referenced_count"]
                if item["referenced_count"]
                else 0.0
            )
            if unresolved_fraction > exclude_unresolved_fraction_above:
                excluded_by_unresolved_sensitivity += 1
            else:
                qualified_ground_truth.append(item)
    else:
        qualified_ground_truth = qualified_before_sensitivity
    relevant_by_survey = {
        item["survey_id"]: set(item["relevant_candidate_ids"])
        for item in qualified_ground_truth
    }
    _write_jsonl(output_dir / "ground_truth.jsonl", ground_truth)

    source_metadata.extend(
        _metadata_record(
            record,
            origin=(
                "OpenAlex OR-filter list API"
                if _canonical_id(record["id"]) in resolved_references
                else "Phase-1 OpenAlex corpus"
            ),
            status=(
                "resolved"
                if _canonical_id(record["id"]) in resolved_references
                else "corpus record"
            ),
        )
        for record in candidates
    )
    _write_jsonl(output_dir / "source_metadata.jsonl", source_metadata)
    corpus_documents = [document_text(record) for record in candidates]
    bm25 = BM25Index.from_documents(corpus_documents)
    rankings: dict[str, dict[str, list[str]]] = {
        method: {}
        for method in ("BM25", "dense", "RRF", "weighted_RRF_LOO")
    }
    scores: dict[str, dict[str, dict[str, float]]] = {
        method: {}
        for method in ("BM25", "dense", "RRF", "weighted_RRF_LOO")
    }
    bm25_trials: list[dict[str, Any]] = []
    dense_status = "not_run"
    dense = DenseRetriever(model, cache_dir=config.CACHE_DIR / "retrieval")
    try:
        dense.fit(corpus_documents)
        dense_status = "available"
        dense_backend = "faiss.IndexFlatIP (explicit)" if dense.used_faiss else "numpy_exact_cosine (default)"
    except Exception as exc:
        dense_status = f"unavailable: {type(exc).__name__}"
        dense_backend = None

    dense_trials: list[dict[str, Any]] = []
    for item in qualified_ground_truth:
        sid = item["survey_id"]
        query = str(item.get("query") or "")
        bm25_order = [
            pair
            for pair in bm25.rank(query)
            if candidate_ids[pair[0]] != sid
        ]
        bm25_ids = [candidate_ids[index] for index, _score in bm25_order]
        rankings["BM25"][sid] = bm25_ids
        scores["BM25"][sid] = {
            candidate_ids[index]: float(score) for index, score in bm25_order
        }
        bm25_trials.append(
            {
                "survey_id": sid,
                "query": query,
                "method": "BM25",
                "candidate_count": len(bm25_ids),
                "ranking": [
                    {"work_id": work_id, "score": score, "relevant": work_id in relevant_by_survey[sid]}
                    for work_id, score in list(scores["BM25"][sid].items())[:50]
                ],
            }
        )
        if dense_status == "available":
            try:
                dense_order = [
                    pair
                    for pair in dense.rank(query)
                    if candidate_ids[pair[0]] != sid
                ]
                dense_ids = [candidate_ids[index] for index, _score in dense_order]
                rankings["dense"][sid] = dense_ids
                scores["dense"][sid] = {
                    candidate_ids[index]: float(score) for index, score in dense_order
                }
                rrf_order = reciprocal_rank_fusion(
                    {
                        "BM25": bm25_ids,
                        "dense": dense_ids,
                    },
                    k=RRF_K,
                )
                rankings["RRF"][sid] = [work_id for work_id, _score in rrf_order]
                scores["RRF"][sid] = dict(rrf_order)
                dense_trials.extend(
                    [
                        {
                            "survey_id": sid,
                            "query": query,
                            "method": "dense",
                            "candidate_count": len(dense_ids),
                            "ranking": [
                                {"work_id": work_id, "score": score, "relevant": work_id in relevant_by_survey[sid]}
                                for work_id, score in list(scores["dense"][sid].items())[:50]
                            ],
                        },
                        {
                            "survey_id": sid,
                            "query": query,
                            "method": "RRF",
                            "candidate_count": len(rrf_order),
                            "ranking": [
                                {"work_id": work_id, "score": score, "relevant": work_id in relevant_by_survey[sid]}
                                for work_id, score in rrf_order[:50]
                            ],
                        },
                    ]
                )
            except Exception as exc:
                dense_status = f"unavailable: {type(exc).__name__}"
                rankings["dense"].clear()
                rankings["RRF"].clear()
                rankings["weighted_RRF_LOO"].clear()
                scores["dense"].clear()
                scores["RRF"].clear()
                scores["weighted_RRF_LOO"].clear()
                dense_trials = []
                break

    loo_tuning: list[dict[str, Any]] = []
    if dense_status == "available" and len(qualified_ground_truth) > 1:
        for held_out in qualified_ground_truth:
            held_out_id = held_out["survey_id"]
            training = [
                item
                for item in qualified_ground_truth
                if item["survey_id"] != held_out_id
            ]
            tuned: list[tuple[float, float]] = []
            for dense_weight in WEIGHTED_RRF_DENSE_WEIGHTS:
                training_ndcg = []
                for item in training:
                    sid = item["survey_id"]
                    ranking = [
                        work_id
                        for work_id, _score in reciprocal_rank_fusion(
                            {
                                "BM25": rankings["BM25"][sid],
                                "dense": rankings["dense"][sid],
                            },
                            k=RRF_K,
                            weights={"BM25": 1.0, "dense": dense_weight},
                        )
                    ]
                    value = ndcg_at_k(ranking, relevant_by_survey[sid], 20)
                    if value is not None:
                        training_ndcg.append(value)
                tuned.append(
                    (
                        sum(training_ndcg) / len(training_ndcg)
                        if training_ndcg
                        else float("-inf"),
                        dense_weight,
                    )
                )
            best_training_score, best_dense_weight = max(
                tuned, key=lambda item: (item[0], -item[1])
            )
            held_out_ranking = reciprocal_rank_fusion(
                {
                    "BM25": rankings["BM25"][held_out_id],
                    "dense": rankings["dense"][held_out_id],
                },
                k=RRF_K,
                weights={"BM25": 1.0, "dense": best_dense_weight},
            )
            rankings["weighted_RRF_LOO"][held_out_id] = [
                work_id for work_id, _score in held_out_ranking
            ]
            scores["weighted_RRF_LOO"][held_out_id] = dict(held_out_ranking)
            loo_tuning.append(
                {
                    "survey_id": held_out_id,
                    "dense_weight": best_dense_weight,
                    "bm25_weight": 1.0,
                    "training_queries": len(training),
                    "training_ndcg_at_20": best_training_score,
                    "selection_metric": "mean nDCG@20 on other queries",
                    "k": RRF_K,
                }
            )
            dense_trials.append(
                {
                    "survey_id": held_out_id,
                    "query": held_out.get("query"),
                    "method": "weighted_RRF_LOO",
                    "role": "ablation_only",
                    "candidate_count": len(held_out_ranking),
                    "loo_tuned_weights": {
                        "BM25": 1.0,
                        "dense": best_dense_weight,
                    },
                    "ranking": [
                        {
                            "work_id": work_id,
                            "score": score,
                            "relevant": work_id in relevant_by_survey[held_out_id],
                        }
                        for work_id, score in held_out_ranking[:50]
                    ],
                }
            )

    _write_jsonl(output_dir / "trials_bm25.jsonl", bm25_trials)
    _write_jsonl(output_dir / "trials_dense_rrf.jsonl", dense_trials)
    _write_jsonl(output_dir / "candidate_pool.jsonl", candidates)
    _write_jsonl(
        output_dir / "unresolved_reference_ids.jsonl",
        (
            {
                "id": f"https://openalex.org/{work_id}",
                "resolution_status": (
                    "no work record returned by OpenAlex OR-filter or singleton lookup"
                ),
            }
            for work_id in sorted(unresolved_relevant_ids)
        ),
    )
    _write_jsonl(
        output_dir / "unresolved_singleton_recheck.jsonl",
        secondary_recheck_rows,
    )
    _write_csv(
        output_dir / "unresolved_by_survey.csv",
        [
            "survey_id",
            "total_refs",
            "unresolved",
            "fraction",
            "resolution_status",
        ],
        (
            {
                "survey_id": item["survey_id"],
                "total_refs": item["referenced_count"],
                "unresolved": (
                    len(item["unresolved_reference_ids"])
                    if item["source_abstract_available"]
                    else ""
                ),
                "fraction": (
                    len(item["unresolved_reference_ids"]) / item["referenced_count"]
                    if item["source_abstract_available"]
                    and item["referenced_count"]
                    else (
                        0.0
                        if item["source_abstract_available"]
                        else ""
                    )
                ),
                "resolution_status": (
                    "resolved refs evaluated"
                    if item["source_abstract_available"]
                    else "unassessed: survey abstract missing"
                ),
            }
            for item in ground_truth
        ),
    )
    _write_csv(
        output_dir / "survey_qualification.csv",
        [
            "survey_id",
            "survey_title",
            "referenced_count",
            "abstract_relevant_count",
            "minimum_relevant_documents",
            "source_abstract_available",
            "qualified",
            "reason",
        ],
        qualification_rows,
    )
    prisma = {
        "phase1_corpus_records": len(records),
        "phase1_records_without_abstract": len(no_abstract_phase1_ids),
        "phase1_abstract_records": len(records) - len(no_abstract_phase1_ids),
        "survey_candidate_pool_available": survey_candidates_available,
        "survey_queries_screened": screened_survey_count,
        "survey_query_limit": max_surveys,
        "survey_query_slots_unfilled": max(0, max_surveys - screened_survey_count),
        "survey_sources_without_abstract": sum(
            not bool(item["source_abstract_available"]) for item in ground_truth
        ),
        "survey_sources_below_relevant_threshold": sum(
            bool(item["source_abstract_available"])
            and int(item["relevant_candidate_count"]) < min_relevant_documents
            for item in ground_truth
        ),
        "survey_sources_excluded": screened_survey_count - len(qualified_ground_truth),
        "unique_referenced_work_ids_identified": len(all_reference_ids),
        "referenced_records_resolved": len(resolved_references),
        "referenced_records_unresolved": len(unresolved_relevant_ids),
        "singleton_recheck_attempted": len(secondary_recheck_rows),
        "singleton_recheck_resolved": sum(
            row["status"] == "resolved by singleton endpoint"
            for row in secondary_recheck_rows
        ),
        "singleton_recheck_limit": SECONDARY_RECHECK_LIMIT,
        "referenced_records_without_abstract": len(no_abstract_reference_ids),
        "referenced_records_with_abstract": len(resolved_references) - len(no_abstract_reference_ids),
        "referenced_records_added_to_phase1_pool": len(
            set(pool_by_id) - set(by_id)
        ),
        "duplicate_references_already_in_phase1": len(
            set(resolved_references) & set(by_id)
        ),
        "expanded_candidate_pool_included": len(candidates),
        "minimum_relevant_documents_per_survey": min_relevant_documents,
        "qualified_survey_count": len(qualified_ground_truth),
        "qualified_before_unresolved_sensitivity": len(
            qualified_before_sensitivity
        ),
        "sensitivity_unresolved_fraction_threshold": (
            exclude_unresolved_fraction_above
            if exclude_unresolved_fraction_above is not None
            else ""
        ),
        "sensitivity_excluded_survey_count": excluded_by_unresolved_sensitivity,
        "excluded_survey_count": screened_survey_count - len(qualified_ground_truth),
        "excluded_survey_reasons": {
            reason: sum(row["reason"] == reason for row in qualification_rows)
            for reason in sorted({row["reason"] for row in qualification_rows if not row["qualified"]})
        },
        "survey_selection_scope": (
            "top-ranked survey/review titles about graph neural, graph convolutional, "
            "or graph learning works in the union of Phase-1 corpus and resolved references"
        ),
        "survey_selection_order": "publication year descending, OpenAlex work ID ascending",
        "qualification_rule": (
            f"abstract present and at least {min_relevant_documents} cited candidates "
            "with abstracts in the expanded pool"
        ),
    }
    (output_dir / "retrieval_prisma.json").write_text(
        json.dumps(prisma, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metric_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    paired_test_rows: list[dict[str, Any]] = []
    n_relevant = sum(len(ids) for ids in relevant_by_survey.values())
    methods = ("BM25", "dense", "RRF", "weighted_RRF_LOO")
    metric_cutoffs = {
        "Recall": tuple(cutoffs),
        "nDCG": DEFAULT_NDCG_CUTOFFS,
        "Precision": tuple(cutoffs),
        "MAP": (0,),
    }
    method_query_values: dict[tuple[str, str, int], dict[str, float]] = {}
    for method_index, method in enumerate(methods):
        available = method == "BM25" or (
            dense_status == "available"
            and (method != "weighted_RRF_LOO" or len(qualified_ground_truth) > 1)
        )
        method_status = (
            "ok"
            if available
            else (
                "LOO tuning requires at least two qualified queries"
                if method == "weighted_RRF_LOO" and dense_status == "available"
                else dense_status
            )
        )
        for metric_name, cutoffs_for_metric in metric_cutoffs.items():
            for cutoff in cutoffs_for_metric:
                per_query_values: list[float] = []
                values_by_query: dict[str, float] = {}
                if available:
                    for item in qualified_ground_truth:
                        sid = item["survey_id"]
                        if metric_name == "MAP":
                            value = mean_average_precision(
                                rankings[method][sid], relevant_by_survey[sid]
                            )
                        elif metric_name == "Recall":
                            value = recall_at_k(
                                rankings[method][sid],
                                relevant_by_survey[sid],
                                cutoff,
                            )
                        elif metric_name == "nDCG":
                            value = ndcg_at_k(
                                rankings[method][sid],
                                relevant_by_survey[sid],
                                cutoff,
                            )
                        else:
                            value = precision_at_k(
                                rankings[method][sid],
                                relevant_by_survey[sid],
                                cutoff,
                            )
                        per_query_rows.append(
                            {
                                "survey_id": sid,
                                "survey_title": item["survey_title"],
                                "method": method,
                                "metric": metric_name,
                                "k": "" if metric_name == "MAP" else cutoff,
                                "value": value,
                                "relevant_candidate_count": len(relevant_by_survey[sid]),
                                "status": "ok" if value is not None else "no relevant candidates",
                            }
                        )
                        if value is not None:
                            per_query_values.append(value)
                            values_by_query[sid] = value
                mean, ci_lower, ci_upper = _bootstrap_mean_ci(
                    per_query_values,
                    seed=BOOTSTRAP_SEED + method_index * 100 + cutoff,
                )
                method_query_values[(method, metric_name, cutoff)] = values_by_query
                metric_rows.append(
                    {
                        "method": method,
                        "role": "ablation_only" if method == "weighted_RRF_LOO" else "headline",
                        "metric": (
                            "MAP"
                            if metric_name == "MAP"
                            else f"{metric_name}@{cutoff}"
                        ),
                        "k": "" if metric_name == "MAP" else cutoff,
                        "mean": mean if mean is not None else "",
                        "ci_lower_95": ci_lower if ci_lower is not None else "",
                        "ci_upper_95": ci_upper if ci_upper is not None else "",
                        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                        "bootstrap_seed": BOOTSTRAP_SEED + method_index * 100 + cutoff,
                        "n_queries": len(per_query_values),
                        "n_surveys": len(qualified_ground_truth),
                        "n_relevant_ids": n_relevant,
                        "status": method_status,
                    }
                )
    for metric_name, cutoff in (
        (metric, cutoff)
        for metric, values in metric_cutoffs.items()
        for cutoff in values
        if metric in {"Recall", "nDCG"}
    ):
        for left_method, right_method in (("dense", "BM25"), ("dense", "RRF")):
            left_values = method_query_values.get(
                (left_method, metric_name, cutoff), {}
            )
            right_values = method_query_values.get(
                (right_method, metric_name, cutoff), {}
            )
            shared_ids = sorted(set(left_values) & set(right_values))
            p_value = (
                _paired_wilcoxon_pvalue(
                    [left_values[sid] for sid in shared_ids],
                    [right_values[sid] for sid in shared_ids],
                )
                if len(shared_ids) >= 1
                else None
            )
            paired_test_rows.append(
                {
                    "metric": f"{metric_name}@{cutoff}",
                    "left_method": left_method,
                    "right_method": right_method,
                    "n_paired_queries": len(shared_ids),
                    "wilcoxon_p_value_two_sided": p_value if p_value is not None else "",
                    "holm_p_dense_vs_bm25_5": "",
                    "holm_p_all_10": "",
                    "test": "exact paired Wilcoxon signed-rank sign-flip",
                }
            )
    dense_bm25_rows = [
        index
        for index, row in enumerate(paired_test_rows)
        if row["left_method"] == "dense" and row["right_method"] == "BM25"
    ]
    valid_dense_bm25_rows = [
        index
        for index in dense_bm25_rows
        if paired_test_rows[index]["wilcoxon_p_value_two_sided"] != ""
    ]
    raw_p_values = [
        float(paired_test_rows[index]["wilcoxon_p_value_two_sided"])
        for index in valid_dense_bm25_rows
    ]
    for index, adjusted_p in zip(
        valid_dense_bm25_rows, _holm_adjust(raw_p_values)
    ):
        paired_test_rows[index]["holm_p_dense_vs_bm25_5"] = adjusted_p
    valid_all_rows = [
        index
        for index, row in enumerate(paired_test_rows)
        if row["wilcoxon_p_value_two_sided"] != ""
    ]
    all_p_values = [
        float(paired_test_rows[index]["wilcoxon_p_value_two_sided"])
        for index in valid_all_rows
    ]
    for index, adjusted_p in zip(valid_all_rows, _holm_adjust(all_p_values)):
        paired_test_rows[index]["holm_p_all_10"] = adjusted_p

    _write_csv(
        output_dir / "metrics.csv",
        [
            "method",
            "role",
            "metric",
            "k",
            "mean",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_resamples",
            "bootstrap_seed",
            "n_queries",
            "n_surveys",
            "n_relevant_ids",
            "status",
        ],
        metric_rows,
    )
    _write_csv(
        output_dir / "paired_tests.csv",
        [
            "metric",
            "left_method",
            "right_method",
            "n_paired_queries",
            "wilcoxon_p_value_two_sided",
            "holm_p_dense_vs_bm25_5",
            "holm_p_all_10",
            "test",
        ],
        paired_test_rows,
    )
    _write_csv(
        output_dir / "per_query_metrics.csv",
        ["survey_id", "survey_title", "method", "metric", "k", "value", "relevant_candidate_count", "status"],
        per_query_rows,
    )
    for method in methods:
        _write_csv(
            output_dir / f"per_query_{method}.csv",
            ["survey_id", "survey_title", "method", "metric", "k", "value", "relevant_candidate_count", "status"],
            (row for row in per_query_rows if row["method"] == method),
        )
    status = {
        "status": (
            "complete"
            if dense_status == "available" and not unresolved_relevant_ids
            else "partial"
        ),
        "corpus_path": corpus_path.name,
        "corpus_sha256": _stable_hash(corpus_path),
        "source_survey_count": len(qualified_ground_truth),
        "screened_survey_count": screened_survey_count,
        "survey_candidate_pool_available": survey_candidates_available,
        "qualified_survey_count": len(qualified_ground_truth),
        "excluded_survey_count": screened_survey_count - len(qualified_ground_truth),
        "survey_qualification_ratio": (
            f"{len(qualified_ground_truth)}/{screened_survey_count}"
            if screened_survey_count
            else "0/0"
        ),
        "survey_query_limit": max_surveys,
        "minimum_relevant_documents": min_relevant_documents,
        "evaluated_query_count": len(qualified_ground_truth),
        "candidate_count": len(candidates),
        "referenced_candidate_count": len(set(resolved_references) & candidate_id_set),
        "no_abstract_referenced_record_count": len(no_abstract_reference_ids),
        "ground_truth_relevant_candidate_count": n_relevant,
        "total_survey_citation_relations": sum(
            item["referenced_count"] for item in qualified_ground_truth
        ),
        "unique_survey_cited_work_count": len(all_reference_ids),
        "resolved_cited_candidate_count": len(set(resolved_references) & candidate_id_set),
        "unresolved_relevant_reference_count": len(unresolved_relevant_ids),
        "cutoffs": {
            "Recall": list(cutoffs),
            "nDCG": list(DEFAULT_NDCG_CUTOFFS),
            "Precision": list(cutoffs),
            "MAP": "full ranked pool",
        },
        "embedding_model": model,
        "dense_backend": dense_backend,
        "rrf_k": RRF_K,
        "primary_rrf": "unweighted reciprocal rank fusion, k=60",
        "weighted_rrf": {
            "role": "ablation only",
            "tuning": "leave-one-query-out using mean nDCG@20 on remaining queries",
            "dense_weight_grid": list(WEIGHTED_RRF_DENSE_WEIGHTS),
            "fixed_bm25_weight": 1.0,
        },
        "dense_status": dense_status,
        "metric_aggregation": "mean over qualified survey queries; percentile bootstrap 95% CI, 10,000 resamples",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_test": "exact two-sided paired Wilcoxon signed-rank sign-flip",
        "ground_truth_definition": "OpenAlex referenced_works intersected with all abstract-bearing Phase-1 and expanded cited-work candidates, excluding each query survey itself",
        "openalex_reference_resolution": {
            "method": "works list endpoint using filter=ids.openalex:<ID1>|<ID2> (chunks of at most 100)",
            "documentation": "https://help.openalex.org/api/filtering/#addition-or",
            "or_value_limit": 100,
        },
        "prisma": prisma,
        "loo_weighted_rrf": loo_tuning,
        "scope_limitation": "No-abstract and unresolved cited records are excluded from the retrieval pool and counted in retrieval_prisma.json.",
        "artifacts": [
            "source_metadata.jsonl",
            "resolved_references.jsonl",
            "ground_truth.jsonl",
            "candidate_pool.jsonl",
            "trials_bm25.jsonl",
            "trials_dense_rrf.jsonl",
            "survey_qualification.csv",
            "retrieval_prisma.json",
            "per_query_metrics.csv",
            "per_query_BM25.csv",
            "per_query_dense.csv",
            "per_query_RRF.csv",
            "per_query_weighted_RRF_LOO.csv",
            "metrics.csv",
            "paired_tests.csv",
            "unresolved_reference_ids.jsonl",
            "unresolved_singleton_recheck.jsonl",
            "unresolved_by_survey.csv",
        ],
    }
    (output_dir / "run.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-surveys", type=int, default=20)
    parser.add_argument(
        "--k",
        default="10,20,50",
        help="Comma-separated Recall@k cutoffs (nDCG is always reported at 10 and 20).",
    )
    parser.add_argument(
        "--min-relevant-docs",
        type=int,
        default=MIN_RELEVANT_DOCUMENTS,
        help="Minimum abstract-bearing cited documents required to qualify a survey query.",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--exclude-unresolved-fraction-above",
        type=float,
        default=None,
        help="Sensitivity analysis only: exclude survey queries above this unresolved fraction.",
    )
    args = parser.parse_args(argv)
    try:
        cutoffs = tuple(int(part) for part in args.k.split(",") if part.strip())
        result = run_evaluation(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            max_surveys=args.max_surveys,
            min_relevant_documents=args.min_relevant_docs,
            cutoffs=cutoffs,
            model_name=args.embedding_model,
            exclude_unresolved_fraction_above=args.exclude_unresolved_fraction_above,
        )
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
