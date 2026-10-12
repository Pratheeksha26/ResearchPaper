from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone

from gapfinder import config
from gapfinder.config import get_secret_source
from gapfinder.extraction.extractor import (
    ExtractionResult,
    _cache_path,
    _prompt,
    _read_cache,
    extract_batch,
    recover_cached_batch_responses,
    repair_evidence_spans,
)
from gapfinder.extraction.schema import ClaimExtraction, GENERIC_CLAIM_PHRASES
from gapfinder.extraction.normalize import (
    DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    ENTITY_FIELDS,
    normalize_claim_entities,
)
from gapfinder.llm.client import (
    LLMAPIError,
    RequestBudget,
    _is_pre_http_transport_error,
    credential_fingerprint,
)
from gapfinder.config import get_secret
from gapfinder.llm.rate_limiter import (
    PACIFIC,
    daily_attempt_count,
    refund_logged_transport_attempts,
    require_selected_model,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = PACKAGE_ROOT / "corpus.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "experiments" / "runs" / "extraction"
MODEL_STATUS_AS_OF = "2026-09-28"
MODEL_SHUTDOWN_DATE = None
MODEL_DOCS_URL = "https://ai.google.dev/gemini-api/docs/models"
MODEL_DEPRECATION_URL = "https://ai.google.dev/gemini-api/docs/deprecations"
MODEL_THINKING_LEVELS_URL = "https://ai.google.dev/gemini-api/docs/thinking"
STANDARD_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75
COMPARISON_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-4-31b-it",
}
EVIDENCE_FIELDS = (
    "task",
    "method",
    "dataset",
    "metric",
    "setting",
    "stated_limitation",
    "stated_future_work",
)
AUDIT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with",
}


def _read_papers(path: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                paper = {
                    "id": row.get("paper_id"),
                    "title": row.get("title"),
                    "year": row.get("year"),
                    "abstract": row.get("abstract"),
                }
                if str(paper.get("abstract") or "").strip():
                    papers.append(paper)
        return papers
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            paper = json.loads(line)
            if not isinstance(paper, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}.")
            if paper.get("synthetic"):
                raise ValueError("Synthetic records are not eligible for extraction.")
            if str(paper.get("abstract") or "").strip():
                papers.append(paper)
    return papers


def _package_versions() -> dict[str, str | None]:
    packages = (
        "gapfinder",
        "pip",
        "google-genai",
        "pydantic",
        "numpy",
        "requests",
        "sentence-transformers",
        "torch",
        "transformers",
        "ruff",
        "mypy",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _read_usage(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _append_batch_attempt(output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / "batch_attempts.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prior_manifest_count(output_dir: Path, field: str) -> int:
    path = output_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    value = manifest.get(field, 0)
    if not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Run manifest has an invalid {field!r} count.")
    return value


def _reconcile_legacy_transport_attempts(model: str, run_id: str) -> int:
    calls_path = config.RUNS_DIR / run_id / "llm_calls.jsonl"
    if not calls_path.is_file():
        return 0
    today = datetime.now(PACIFIC).date().isoformat()
    attempt_ids: list[str] = []
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        try:
            call = json.loads(line)
            timestamp = call.get("request_timestamp") or call.get("timestamp")
            if not timestamp:
                continue
            request_date = (
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                .astimezone(PACIFIC)
                .date()
                .isoformat()
            )
            if (
                call.get("model") != model
                or request_date != today
                or call.get("status_code") is not None
                or call.get("quota_counted") is False
                or not isinstance(call.get("error"), str)
            ):
                continue
            if _is_pre_http_transport_error(RuntimeError(call["error"])):
                attempt_ids.append(hashlib.sha256(line.encode("utf-8")).hexdigest())
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return refund_logged_transport_attempts(model, attempt_ids)


def _audit_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in AUDIT_STOPWORDS
    ]


def _classify_span(span: str, abstract: str) -> str:
    if span in abstract:
        return "exact"
    span_tokens = _audit_tokens(span)
    abstract_tokens = _audit_tokens(abstract)
    if len(span_tokens) < 3 or not abstract_tokens:
        return "fabricated"
    overlap = len(set(span_tokens) & set(abstract_tokens)) / len(set(span_tokens))
    order_similarity = SequenceMatcher(
        None, span_tokens, abstract_tokens, autojunk=False
    ).ratio()
    return "near_miss" if overlap >= 0.7 or order_similarity >= 0.6 else "fabricated"


def _progress_snapshot(
    papers: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    abstracts = {
        str(paper.get("id") or ""): str(paper.get("abstract") or "").strip()
        for paper in papers
    }
    claims_by_id: dict[
        str,
        tuple[
            dict[str, Any],
            str,
            bool,
            dict[str, Literal["exact", "repaired", "nulled"]],
            dict[str, Any],
        ],
    ] = {}
    for paper in papers:
        paper_id = str(paper.get("id") or "")
        abstract = abstracts[paper_id]
        prompt = _prompt(paper_id, abstract)
        cache_path = _cache_path(
            config.CACHE_DIR / "extraction", paper_id, model, prompt
        )
        cached = _read_cache(cache_path, abstract)
        if cached is not None:
            claims_by_id[paper_id] = (
                cached.claims.model_dump(mode="json"),
                "cache",
                True,
                cached.span_status,
                cached.claims.model_dump(mode="json"),
            )
    calls_path = config.RUNS_DIR / run_id / "llm_calls.jsonl"
    if calls_path.is_file():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            try:
                call = json.loads(line)
                if call.get("model") != model or call.get("status_code") != 200:
                    continue
                response = json.loads(call.get("response") or "")
                for item in response.get("items", []):
                    paper_id = str(item.get("paper_id") or "")
                    if paper_id not in abstracts or paper_id in claims_by_id:
                        continue
                    claims = item.get("claims")
                    if not isinstance(claims, dict):
                        continue
                    try:
                        validated = ClaimExtraction.model_validate(claims)
                        schema_valid = True
                    except Exception:
                        schema_valid = False
                        validated = None
                    statuses: dict[str, Literal["exact", "repaired", "nulled"]] = {}
                    repaired_claims = claims
                    if validated is not None:
                        repaired, statuses = repair_evidence_spans(
                            validated, abstracts[paper_id]
                        )
                        repaired_claims = repaired.model_dump(mode="json")
                    claims_by_id[paper_id] = (
                        repaired_claims,
                        "response",
                        schema_valid,
                        statuses,
                        claims,
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    adjusted_originals: dict[tuple[str, str], str] = {}
    adjustments_path = config.RUNS_DIR / run_id / "span_adjustments.jsonl"
    if adjustments_path.is_file():
        for line in adjustments_path.read_text(encoding="utf-8").splitlines():
            try:
                adjustment = json.loads(line)
                original_span = adjustment.get("original_span")
                if isinstance(original_span, str) and original_span:
                    adjusted_originals[
                        (str(adjustment.get("paper_id") or ""), str(adjustment.get("field") or ""))
                    ] = original_span
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    audit_rows: list[dict[str, Any]] = []
    span_status_counts = {"exact": 0, "repaired": 0, "nulled": 0}
    for paper_id, (claims, source, schema_valid, statuses, raw_claims) in claims_by_id.items():
        for status in statuses.values():
            if status in span_status_counts:
                span_status_counts[status] += 1
        for field in EVIDENCE_FIELDS:
            value = raw_claims.get(field)
            span = adjusted_originals.get((paper_id, field))
            if span is None and isinstance(value, dict):
                span = value.get("evidence_span")
            if not isinstance(span, str) or not span:
                continue
            audit_rows.append(
                {
                    "paper_id": paper_id,
                    "field": field,
                    "classification": _classify_span(span, abstracts[paper_id]),
                    "source": source,
                    "schema_valid": schema_valid,
                    "evidence_span": span,
                    "span_status": statuses.get(field, "model_null"),
                }
            )
    audit_path = output_dir / "evidence_span_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "paper_id",
                "field",
                "classification",
                "source",
                "schema_valid",
                "evidence_span",
                "span_status",
            ),
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    categories = {
        name: sum(row["classification"] == name for row in audit_rows)
        for name in ("exact", "near_miss", "fabricated")
    }
    completed_count = sum(
        source == "cache" and schema_valid
        for _claims, source, schema_valid, _statuses, _raw in claims_by_id.values()
    )
    schema_valid_count = sum(
        schema_valid
        for _claims, _source, schema_valid, _statuses, _raw in claims_by_id.values()
    )
    span_count = len(audit_rows)
    limits = require_selected_model(model)
    api_key = (
        get_secret("GEMINI_API_KEY2")
        or get_secret("GEMINI_API_KEY")
        or get_secret("GAPFINDER_API_KEY")
    )
    quota_bucket = credential_fingerprint(api_key) if api_key else None
    quota_used = daily_attempt_count(model, quota_bucket=quota_bucket)
    return {
        "run_id": run_id,
        "model_id": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "papers_total": len(papers),
        "papers_completed": completed_count,
        "schema_valid_rate": (
            schema_valid_count / len(papers) if papers else None
        ),
        "schema_valid_papers": schema_valid_count,
        "exact_span_rate": categories["exact"] / span_count if span_count else None,
        "exact_spans": categories["exact"],
        "span_adjustment_counts": span_status_counts,
        "exact_span_status_rate": (
            span_status_counts["exact"] / sum(span_status_counts.values())
            if sum(span_status_counts.values())
            else None
        ),
        "repaired_span_rate": (
            span_status_counts["repaired"] / sum(span_status_counts.values())
            if sum(span_status_counts.values())
            else None
        ),
        "nulled_span_rate": (
            span_status_counts["nulled"] / sum(span_status_counts.values())
            if sum(span_status_counts.values())
            else None
        ),
        "near_miss_spans": categories["near_miss"],
        "fabricated_spans": categories["fabricated"],
        "fabrication_rate": (
            categories["fabricated"] / span_count if span_count else None
        ),
        "classified_non_null_spans": span_count,
        "papers_remaining": len(papers) - completed_count,
        "quota_attempts_today": quota_used,
        "quota_attempts_remaining": max(0, limits.rpd - quota_used),
        "span_classification_method": (
            "Exact raw substring; otherwise near-miss when >=70% of informative "
            "tokens overlap or token-order similarity is >=0.60; otherwise fabricated."
        ),
    }


def run_extraction(
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    output_dir: Path = DEFAULT_OUTPUT,
    model: str | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    pilot: bool = False,
    full_approved: bool = False,
    comparison_approved: bool = False,
    max_requests: int | None = None,
    run_id_override: str | None = None,
) -> dict[str, Any]:
    """Run extraction and persist results, per-paper outcomes, and model manifest."""
    if not pilot and not full_approved and not comparison_approved:
        raise RuntimeError(
            "Extraction requires explicit pilot, comparison, or full-run approval."
        )
    if comparison_approved and (
        pilot
        or full_approved
        or corpus_path.name != "extraction_sample.csv"
        or limit is not None
    ):
        raise ValueError(
            "A model comparison must use the complete extraction_sample.csv only."
        )
    if pilot and (limit is not None and limit != 20):
        raise ValueError("Pilot runs are fixed to 20 papers.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive.")
    if pilot and max_requests is None:
        raise ValueError("A capped pilot requires --max-requests.")
    papers = _read_papers(corpus_path)
    full_corpus_count = len(papers)
    if comparison_approved and full_corpus_count != 100:
        raise ValueError(
            f"Model comparison requires exactly 100 sample papers; "
            f"found {full_corpus_count}."
        )
    if pilot:
        limit = 20
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when specified.")
        papers = papers[:limit]
    chosen_model = model or config.MODEL
    if comparison_approved and chosen_model not in COMPARISON_MODELS:
        raise ValueError(
            f"Comparison model must be one of {sorted(COMPARISON_MODELS)}."
        )
    limits = require_selected_model(chosen_model)
    run_id = run_id_override or (
        f"{output_dir.name}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_transport_refunds = _reconcile_legacy_transport_attempts(
        chosen_model, run_id
    )
    recovered_from_raw = recover_cached_batch_responses(
        papers,
        model=chosen_model,
        run_id=run_id,
    )
    progress_path = output_dir / "daily_progress.jsonl"

    def record_daily_progress() -> None:
        snapshot = _progress_snapshot(
            papers,
            model=chosen_model,
            run_id=run_id,
            output_dir=output_dir,
        )
        with progress_path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        print(
            "Quota window paused; "
            f"{snapshot['papers_completed']}/{snapshot['papers_total']} papers "
            f"completed, schema-valid={snapshot['schema_valid_rate']}, "
            f"raw-exact-span={snapshot['exact_span_rate']}, "
            "exact/repaired/nulled rates="
            f"{snapshot['exact_span_status_rate']}/"
            f"{snapshot['repaired_span_rate']}/"
            f"{snapshot['nulled_span_rate']}, "
            f"quota={snapshot['quota_attempts_today']}/"
            f"{snapshot['quota_attempts_today'] + snapshot['quota_attempts_remaining']}, "
            f"fabricated={snapshot['fabricated_spans']}/"
            f"{snapshot['classified_non_null_spans']}, "
            f"remaining={snapshot['papers_remaining']}."
        )

    request_budget = RequestBudget(
        max_requests,
        on_daily_pause=record_daily_progress,
    )
    thinking_supported_by_documented_gemini_api = chosen_model.startswith("gemini-3.")
    thinking_control = (
        "Gemini 3 thinking_level=low (minimum documented level)"
        if thinking_supported_by_documented_gemini_api
        else "not sent; Google API documentation for this Gemma model is unverified"
    )
    manifest = {
        "run_id": run_id,
        "model_id": chosen_model,
        "api_key_source": (
            get_secret_source("GEMINI_API_KEY2")
            if get_secret_source("GEMINI_API_KEY2") != "missing"
            else (
                get_secret_source("GEMINI_API_KEY")
                if get_secret_source("GEMINI_API_KEY") != "missing"
                else get_secret_source("GAPFINDER_API_KEY")
            )
        ),
        "recovered_cache_entries_from_logged_responses": (
            recovered_from_raw
            + _prior_manifest_count(
                output_dir, "recovered_cache_entries_from_logged_responses"
            )
        ),
        "legacy_transport_attempts_refunded": (
            legacy_transport_refunds
            + _prior_manifest_count(output_dir, "legacy_transport_attempts_refunded")
        ),
        "model_status_as_of": MODEL_STATUS_AS_OF,
        "model_deprecation_status": (
            "not listed as deprecated in the checked model/deprecation documentation"
        ),
        "shutdown_date": MODEL_SHUTDOWN_DATE,
        "model_status_source": MODEL_DOCS_URL,
        "deprecation_schedule_source": MODEL_DEPRECATION_URL,
        "retired_prior_model": {
            "model_id": "gemini-2.5-flash",
            "docs_shutdown_date": None,
            "runtime_status": (
                "Gemini API returned HTTP 404 NOT_FOUND: model unavailable to new users"
            ),
        },
        "temperature": 0.0,
        "thinking_level": "low" if thinking_supported_by_documented_gemini_api else None,
        "thinking_control": thinking_control,
        "thinking_level_source": MODEL_THINKING_LEVELS_URL,
        "structured_output": thinking_supported_by_documented_gemini_api,
        "structured_output_note": (
            "Gemini structured output is requested via response_format."
            if thinking_supported_by_documented_gemini_api
            else "Gemma API structured output support is unverified; use strict JSON-in-prompt validation."
        ),
        "schema": "ClaimExtraction JSON Schema",
        "specificity_stoplist": sorted(GENERIC_CLAIM_PHRASES),
        "usage_csv": "usage.csv",
        "rate_limit": {
            "requests_per_minute": limits.rpm,
            "tokens_per_minute": limits.tpm,
            "requests_per_pacific_day": limits.rpd,
            "max_api_attempts": 5,
            "retry_after_honored": True,
            "exponential_backoff_with_jitter": True,
            "pre_http_transport_failures_refunded": True,
            "request_budget_scope": "model plus one-way API-key fingerprint",
            "sdk_retries_disabled": True,
            "budget_state_path": str(
                os.getenv(
                    "GAPFINDER_LLM_BUDGET_PATH",
                    config.CACHE_DIR / "gemini_request_budget.json",
                )
            ),
        },
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "full_corpus_approval": (
            "user-approved full extraction of corrected corpus_main_v4.jsonl"
            if full_approved
            else "not approved"
        ),
        "comparison_approval": (
            "user-approved 100-paper comparison only"
            if comparison_approved
            else None
        ),
        "pilot_limit": 20 if pilot else None,
        "max_requests": max_requests,
        "batch_size": batch_size,
        "request_attempts": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results: list[ExtractionResult] = []
    failures: list[dict[str, str]] = []
    for batch_number, batch_start in enumerate(
        range(0, len(papers), batch_size), start=1
    ):
        batch = papers[batch_start : batch_start + batch_size]
        attempts_before = request_budget.attempts
        try:
            batch_results, batch_failures = extract_batch(
                batch,
                model=chosen_model,
                run_id=run_id,
                request_budget=request_budget,
            )
        except RuntimeError as exc:
            _append_batch_attempt(
                output_dir,
                {
                    "batch_number": batch_number,
                    "phase": "primary",
                    "paper_ids": [str(paper.get("id") or "") for paper in batch],
                    "successful_ids": [],
                    "failures": {
                        str(paper.get("id") or ""): str(exc) for paper in batch
                    },
                    "request_attempts": request_budget.attempts - attempts_before,
                    "failure_kind": (
                        "transport_error"
                        if isinstance(exc, LLMAPIError)
                        and exc.status_code is None
                        else (
                        "api_error"
                        if isinstance(exc, LLMAPIError)
                        else "request_not_sent_or_run_stopped"
                        )
                    ),
                },
            )
            failures.extend(
                {
                    "paper_id": str(paper.get("id") or ""),
                    "failure_mode": str(exc),
                }
                for paper in batch
            )
            if max_requests is not None and request_budget.attempts >= max_requests:
                break
            if isinstance(exc, LLMAPIError) and exc.status_code is None:
                break
            if not full_approved and not pilot and not comparison_approved:
                raise
            if full_approved:
                for paper in batch:
                    paper_id = str(paper.get("id") or "")
                    single_attempts_before = request_budget.attempts
                    try:
                        retried, retry_failures = extract_batch(
                            [paper],
                            model=chosen_model,
                            run_id=run_id,
                            request_budget=request_budget,
                        )
                    except RuntimeError as single_exc:
                        failure_mode = str(single_exc)
                        retried = []
                        retry_failures = {paper_id: failure_mode}
                    if retried:
                        results.extend(retried)
                        failures = [
                            row for row in failures if row["paper_id"] != paper_id
                        ]
                    else:
                        failures.append(
                            {
                                "paper_id": paper_id,
                                "failure_mode": retry_failures.get(
                                    paper_id, str(exc)
                                ),
                            }
                        )
                    _append_batch_attempt(
                        output_dir,
                        {
                            "batch_number": batch_number,
                            "phase": "single_paper_requeue_after_batch_error",
                            "paper_ids": [paper_id],
                            "successful_ids": [row.paper_id for row in retried],
                            "failures": retry_failures,
                            "request_attempts": (
                                request_budget.attempts - single_attempts_before
                            ),
                        },
                    )
                    if (
                        request_budget.maximum is not None
                        and request_budget.attempts >= request_budget.maximum
                    ):
                        break
            if pilot:
                break
            if full_approved:
                continue
            continue
        results.extend(batch_results)
        _append_batch_attempt(
            output_dir,
            {
                "batch_number": batch_number,
                "phase": "primary",
                "paper_ids": [str(paper.get("id") or "") for paper in batch],
                "successful_ids": [result.paper_id for result in batch_results],
                "failures": batch_failures,
                "request_attempts": request_budget.attempts - attempts_before,
            },
        )
        for paper_id, failure_mode in batch_failures.items():
            failures.append({"paper_id": paper_id, "failure_mode": failure_mode})
        if batch_failures:
            for paper_id, failure_mode in list(batch_failures.items()):
                paper_or_none: dict[str, Any] | None = next(
                    (
                        row
                        for row in batch
                        if str(row.get("id") or "") == paper_id
                    ),
                    None,
                )
                if paper_or_none is None:
                    continue
                paper = paper_or_none
                attempts_before = request_budget.attempts
                try:
                    retried, retry_failures = extract_batch(
                        [paper],
                        model=chosen_model,
                        run_id=run_id,
                        request_budget=request_budget,
                    )
                except RuntimeError as exc:
                    _append_batch_attempt(
                        output_dir,
                        {
                            "batch_number": batch_number,
                            "phase": "single_paper_requeue",
                            "paper_ids": [paper_id],
                            "successful_ids": [],
                            "failures": {paper_id: str(exc)},
                            "request_attempts": (
                                request_budget.attempts - attempts_before
                            ),
                            "failure_kind": (
                                "api_error"
                                if isinstance(exc, LLMAPIError)
                                else "request_not_sent_or_run_stopped"
                            ),
                        },
                    )
                    if (
                        request_budget.maximum is not None
                        and request_budget.attempts >= request_budget.maximum
                    ):
                        break
                    failures.append(
                        {"paper_id": paper_id, "failure_mode": str(exc)}
                    )
                    continue
                _append_batch_attempt(
                    output_dir,
                    {
                        "batch_number": batch_number,
                        "phase": "single_paper_requeue",
                        "paper_ids": [paper_id],
                        "successful_ids": [result.paper_id for result in retried],
                        "failures": retry_failures,
                        "request_attempts": request_budget.attempts - attempts_before,
                    },
                )
                if retried:
                    results.extend(retried)
                    failures = [
                        row for row in failures if row["paper_id"] != paper_id
                    ]
                elif retry_failures:
                    failures = [
                        row for row in failures if row["paper_id"] != paper_id
                    ]
                    failures.append(
                        {
                            "paper_id": paper_id,
                            "failure_mode": retry_failures[paper_id],
                        }
                    )
            if (
                request_budget.maximum is not None
                and request_budget.attempts >= request_budget.maximum
            ):
                break
        if failures and not full_approved and not pilot and not comparison_approved:
            raise RuntimeError(f"Batch extraction failed: {failures[0]}")
        if (
            request_budget.maximum is not None
            and request_budget.attempts >= request_budget.maximum
        ):
            break
    entity_terms: dict[str, dict[str, str]] = {}
    if results and not pilot and not comparison_approved:
        from gapfinder.retrieval.dense import DenseRetriever, configured_model_name

        entity_terms = normalize_claim_entities(
            [result.claims for result in results],
            DenseRetriever(
                configured_model_name(),
                cache_dir=config.CACHE_DIR / "extraction_entities",
            ).encoder,
            similarity_threshold=DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
        )
    result_ids = {result.paper_id for result in results}
    failure_by_id = {row["paper_id"]: row for row in failures}
    for paper in papers:
        paper_id = str(paper.get("id") or "")
        if paper_id in result_ids or paper_id in failure_by_id:
            continue
        reason = (
            f"Not processed: run request cap of {max_requests} was reached."
            if max_requests is not None
            and request_budget.attempts >= max_requests
            else "Not processed before the run stopped."
        )
        failure_by_id[paper_id] = {
            "paper_id": paper_id,
            "failure_mode": reason,
        }
    failures = list(failure_by_id.values())
    canonical_surface_count = sum(len(mapping) for mapping in entity_terms.values())
    canonical_target_count = sum(
        len(set(mapping.values())) for mapping in entity_terms.values()
    )
    canonical_artifact = {
        "schema_version": 2,
        "embedding_model": (
            configured_model_name()
            if results and not pilot and not comparison_approved
            else None
        ),
        "similarity_threshold": DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
        "algorithm": "field-specific complete-linkage agglomeration",
        "fields": entity_terms,
    }
    manifest["canonical_entity_normalization"] = {
        "schema_version": 2,
        "embedding_model": canonical_artifact["embedding_model"],
        "similarity_threshold": DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
        "algorithm": canonical_artifact["algorithm"],
        "field_count": len(ENTITY_FIELDS),
        "surface_form_count": canonical_surface_count,
        "canonical_target_count_within_fields": canonical_target_count,
    }
    (output_dir / "canonical_entities.json").write_text(
        json.dumps(canonical_artifact, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_path = output_dir / "extractions.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as target:
        for result in results:
            target.write(result.model_dump_json() + "\n")
    outcomes_path = output_dir / "paper_outcomes.jsonl"
    with outcomes_path.open("w", encoding="utf-8", newline="\n") as target:
        for result in results:
            target.write(
                json.dumps(
                    {
                        "paper_id": result.paper_id,
                        "schema_valid": True,
                        "failure_mode": None,
                        "paper_type": result.claims.paper_type,
                        "include_in_counts": result.claims.include_in_counts,
                    }
                )
                + "\n"
            )
        for failure in failures:
            not_processed = failure["failure_mode"].startswith("Not processed:")
            target.write(
                json.dumps(
                    {
                        **failure,
                        "schema_valid": None if not_processed else False,
                        "status": "not_processed" if not_processed else "invalid",
                    }
                )
                + "\n"
            )
    abstracts_by_id = {
        str(paper.get("id")): str(paper.get("abstract") or "")
        for paper in papers
    }
    evidence_spans = [
        (result.paper_id, getattr(result.claims, field).evidence_span)
        for result in results
        for field in (
            "task",
            "method",
            "dataset",
            "metric",
            "setting",
            "stated_limitation",
            "stated_future_work",
        )
        if getattr(result.claims, field).evidence_span is not None
    ]
    exact_span_count = sum(
        span in abstracts_by_id[paper_id]
        for paper_id, span in evidence_spans
        if span is not None
    )
    span_total = len(evidence_spans)
    usage_rows = _read_usage(config.RUNS_DIR / run_id / "usage.csv")
    actual_input_tokens = sum(
        int(row["input_tokens"] or 0)
        for row in usage_rows
        if row.get("input_tokens", "").isdigit()
    )
    actual_output_tokens = sum(
        int(row["output_tokens"] or 0)
        for row in usage_rows
        if row.get("output_tokens", "").isdigit()
    )
    actual_thinking_tokens = sum(
        int(row["thinking_tokens"] or 0)
        for row in usage_rows
        if row.get("thinking_tokens", "").isdigit()
    )
    usage_metadata_count = sum(
        bool(row.get("input_tokens")) or bool(row.get("output_tokens"))
        for row in usage_rows
    )
    specificity_rejection_path = (
        config.RUNS_DIR / run_id / "specificity_rejections.csv"
    )
    specificity_rejections: list[dict[str, str]] = []
    if specificity_rejection_path.is_file():
        with specificity_rejection_path.open(
            encoding="utf-8", newline=""
        ) as source:
            specificity_rejections = list(csv.DictReader(source))
    specificity_rejected_papers = {
        row["paper_id"] for row in specificity_rejections if row.get("paper_id")
    }
    attempted_count = len(papers)
    schema_valid_rate = len(results) / attempted_count if attempted_count else None
    schema_failure_count = sum(
        "failed validation" in row["failure_mode"].casefold()
        for row in failures
    )
    not_processed_count = sum(
        row["failure_mode"].startswith("Not processed:")
        for row in failures
    )
    manifest["request_attempts"] = len(usage_rows)
    manifest["request_attempts_this_process"] = request_budget.attempts
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "status": (
            "pilot_complete"
            if pilot and not failures and len(results) == len(papers)
            else "pilot_partial"
            if pilot
            else "comparison_complete"
            if comparison_approved and not failures
            else "comparison_partial"
            if comparison_approved
            else "complete"
            if full_approved and not failures
            else "partial"
            if full_approved
            else "complete"
        ),
        "run_id": run_id,
        "model_id": chosen_model,
        "input_corpus": corpus_path.name,
        "papers_extracted": len(results),
        "pilot_attempted": attempted_count,
        "pilot_selected_count": attempted_count,
        "pilot_valid_count": len(result_ids),
        "pilot_planned": len(papers) if pilot else None,
        "schema_valid_rate": schema_valid_rate,
        "exact_evidence_span_rate": (
            exact_span_count / span_total if span_total else None
        ),
        "missing_paper_count": attempted_count - len(results),
        "missing_paper_rate": (
            (attempted_count - len(results)) / attempted_count
            if attempted_count
            else None
        ),
        "non_null_evidence_spans": span_total,
        "non_null_spans_exact_substring": exact_span_count,
        "schema_failures": schema_failure_count,
        "not_processed_count": not_processed_count,
        "failure_modes": failures,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
        "actual_thinking_tokens": actual_thinking_tokens,
        "actual_billed_output_tokens": actual_output_tokens + actual_thinking_tokens,
        "actual_total_tokens": (
            actual_input_tokens + actual_output_tokens + actual_thinking_tokens
        ),
        "api_call_count": len(usage_rows),
        "calls_with_usage_metadata": usage_metadata_count,
        "token_usage_source": "Gemini usage_metadata saved in usage.csv",
        "specificity_rejections": len(specificity_rejections),
        "specificity_rejection_rows": specificity_rejections,
        "papers_rejected_by_specificity": len(specificity_rejected_papers),
        "specificity_rejected_paper_fraction": (
            len(specificity_rejected_papers) / attempted_count
            if attempted_count
            else None
        ),
        "span_validation": "every non-null evidence span is an exact abstract substring",
        "results_file": output_path.name,
        "outcomes_file": outcomes_path.name,
        "canonical_entity_count": canonical_surface_count,
        "canonical_entity_target_count_within_fields": canonical_target_count,
        "canonical_entity_normalization": manifest[
            "canonical_entity_normalization"
        ],
        "paper_type_distribution": {
            paper_type: sum(
                result.claims.paper_type == paper_type for result in results
            )
            for paper_type in (
                "survey_or_review",
                "new_method",
                "benchmark_or_dataset",
                "empirical_study",
                "application",
                "position_or_other",
            )
        },
    }
    progress = _progress_snapshot(
        papers,
        model=chosen_model,
        run_id=run_id,
        output_dir=output_dir,
    )
    summary["evidence_span_audit"] = {
        key: progress[key]
        for key in (
            "exact_spans",
            "near_miss_spans",
            "fabricated_spans",
            "classified_non_null_spans",
            "fabrication_rate",
            "span_classification_method",
        )
    }
    summary["daily_progress_file"] = progress_path.name
    summary["evidence_span_audit_file"] = "evidence_span_audit.csv"
    if pilot:
        example_results = list(results)
        survey_example = next(
            (
                result
                for result in example_results
                if result.claims.paper_type == "survey_or_review"
            ),
            None,
        )
        method_example = next(
            (
                result
                for result in example_results
                if result.claims.paper_type == "new_method"
            ),
            None,
        )
        ordered_examples = [
            result
            for result in (survey_example, method_example, *example_results)
            if result is not None
        ]
        seen_example_ids: set[str] = set()
        examples: list[dict[str, Any]] = []
        for result in ordered_examples:
            if result.paper_id in seen_example_ids or len(examples) >= 5:
                continue
            seen_example_ids.add(result.paper_id)
            source_paper = next(
                (
                    paper
                    for paper in papers
                    if str(paper.get("id") or "") == result.paper_id
                ),
                {},
            )
            examples.append(
                {
                    "paper_id": result.paper_id,
                    "title": source_paper.get("title") or "",
                    "year": source_paper.get("year"),
                    "claims": result.claims.model_dump(mode="json"),
                }
            )
        (output_dir / "pilot_examples.json").write_text(
            json.dumps(examples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["examples_file"] = "pilot_examples.json"
        if results:
            scale = full_corpus_count / len(results)
            estimated_total = (
                actual_input_tokens * scale / 1_000_000 * INPUT_PRICE_PER_MILLION
                + (actual_output_tokens + actual_thinking_tokens)
                * scale
                / 1_000_000
                * OUTPUT_PRICE_PER_MILLION
            )
            summary["scaled_cost_estimate_usd"] = {
                "estimated_input_usd": (
                    actual_input_tokens * scale / 1_000_000 * INPUT_PRICE_PER_MILLION
                ),
                "estimated_output_usd": (
                    (actual_output_tokens + actual_thinking_tokens)
                    * scale
                    / 1_000_000
                    * OUTPUT_PRICE_PER_MILLION
                ),
                "estimated_total_usd": estimated_total,
                "two_times_safety_margin_usd": estimated_total * 2,
                "full_corpus_records": full_corpus_count,
                "pilot_successes_used_for_scaling": len(results),
                "assumption": (
                    "linear scaling from successful pilot actual Gemini usage including "
                    "thinking tokens; excludes retries and cache effects; estimate only, "
                    "not approval to run full extraction"
                ),
                "standard_tier_price_usd_per_million_tokens": {
                    "input": INPUT_PRICE_PER_MILLION,
                    "output": OUTPUT_PRICE_PER_MILLION,
                    "source": STANDARD_PRICING_URL,
                    "effective_through": "2026-12-31",
                },
            }
        else:
            summary["scaled_cost_estimate_usd"] = None
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract evidence-grounded claims from the corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pilot", action="store_true", help="Run only the fixed 20-paper Phase 3 pilot.")
    parser.add_argument(
        "--full-approved",
        action="store_true",
        help="Explicitly acknowledge user approval before a full-corpus run.",
    )
    parser.add_argument(
        "--comparison-approved",
        action="store_true",
        help="Run only the approved 100-paper extraction_sample.csv comparison.",
    )
    args = parser.parse_args()
    summary = run_extraction(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        pilot=args.pilot,
        full_approved=args.full_approved,
        comparison_approved=args.comparison_approved,
        max_requests=args.max_requests,
        run_id_override=args.run_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
