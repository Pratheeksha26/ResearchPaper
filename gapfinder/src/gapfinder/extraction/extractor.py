from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from csv import DictWriter
from collections.abc import Iterable, Iterator, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gapfinder import config
from gapfinder.extraction.schema import (
    ClaimExtraction,
    SpecificityValidationError,
    validate_evidence_spans,
)
from gapfinder.llm.client import LLMAPIError, RequestBudget, generate

EXTRACTION_INSTRUCTIONS = """Extract only information explicitly stated in the abstract.
For each field, return a concise value and copy evidence_span as an exact,
unmodified substring of the given abstract, character-for-character. Do not
normalize whitespace or punctuation, change casing, reorder words, or omit words.
Example: Given text "We evaluate the model on Cora.", the correct evidence_span is
"evaluate the model on Cora" (copied exactly as it appears). If the abstract does
not state a field, set both value and evidence_span to null. Do not infer, expand
abbreviations, or use outside knowledge.
Classify paper_type as survey_or_review, new_method, benchmark_or_dataset,
empirical_study, application, or position_or_other. For survey_or_review, set
task, method, and dataset to null and include_in_counts to false; otherwise set
include_in_counts true. Task, method, and dataset must be specific named entities,
not generic phrases. State substantive limitation/future-work content, not a
pointer to a list of issues or research directions."""


class ExtractionResult(BaseModel):
    """Persisted model output and provenance for one paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    model: str
    prompt_sha256: str
    claims: ClaimExtraction
    span_status: dict[str, Literal["exact", "repaired", "nulled"]] = Field(
        default_factory=dict
    )


def _prompt(paper_id: str, abstract: str) -> str:
    return (
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        f"Paper ID: {paper_id}\n"
        f"Abstract:\n{abstract}"
    )


def _prompt_sha256(prompt: str) -> str:
    fingerprint = json.dumps(
        {"prompt": prompt, "schema": ClaimExtraction.model_json_schema()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _batch_response_schema() -> dict[str, Any]:
    claims_schema = ClaimExtraction.model_json_schema()
    definitions = claims_schema.pop("$defs", {})
    response_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "claims": claims_schema,
                    },
                    "required": ["paper_id", "claims"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    if definitions:
        response_schema["$defs"] = definitions
    return response_schema


def _cache_path(cache_dir: Path, paper_id: str, model: str, prompt: str) -> Path:
    prompt_hash = _prompt_sha256(prompt)
    key = hashlib.sha256(
        f"{paper_id}\0{model}\0{prompt_hash}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.json"


def _read_cache(path: Path, abstract: str) -> ExtractionResult | None:
    if not path.is_file():
        return None
    cached = ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
    if cached.span_status:
        validate_evidence_spans(cached.claims, abstract)
    else:
        claims, statuses = repair_evidence_spans(cached.claims, abstract)
        cached.claims = claims
        cached.span_status = statuses
        path.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
    return cached


_SPAN_FIELDS = (
    "task",
    "method",
    "dataset",
    "metric",
    "setting",
    "stated_limitation",
    "stated_future_work",
)


def _word_tokens(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group().casefold(), match.start(), match.end())
        for match in re.finditer(r"\w+", text, flags=re.UNICODE)
    ]


def repair_evidence_spans(
    claims: ClaimExtraction, abstract: str
) -> tuple[ClaimExtraction, dict[str, Literal["exact", "repaired", "nulled"]]]:
    """Preserve exact spans, snap close spans to literal text, otherwise null the field."""
    updates: dict[str, dict[str, str | None]] = {}
    statuses: dict[str, Literal["exact", "repaired", "nulled"]] = {}
    abstract_tokens = _word_tokens(abstract)
    for field_name in _SPAN_FIELDS:
        evidence = getattr(claims, field_name)
        span = evidence.evidence_span
        if span is None:
            continue
        if span in abstract:
            statuses[field_name] = "exact"
            continue
        target_tokens = [token for token, _start, _end in _word_tokens(span)]
        best: tuple[float, str] | None = None
        if target_tokens and abstract_tokens:
            min_window = max(1, len(target_tokens) - 2)
            max_window = min(len(abstract_tokens), len(target_tokens) + 2)
            for start in range(len(abstract_tokens)):
                for size in range(min_window, max_window + 1):
                    end = start + size
                    if end > len(abstract_tokens):
                        continue
                    candidate = abstract[
                        abstract_tokens[start][1] : abstract_tokens[end - 1][2]
                    ]
                    candidate_tokens = [
                        token for token, _left, _right in abstract_tokens[start:end]
                    ]
                    similarity = SequenceMatcher(
                        None, target_tokens, candidate_tokens, autojunk=False
                    ).ratio()
                    if best is None or similarity > best[0]:
                        best = similarity, candidate
        minimum_similarity = 1.0 if len(target_tokens) <= 2 else 0.8
        if best is not None and best[0] >= minimum_similarity:
            updates[field_name] = {
                "value": evidence.value,
                "evidence_span": best[1],
            }
            statuses[field_name] = "repaired"
        else:
            updates[field_name] = {"value": None, "evidence_span": None}
            statuses[field_name] = "nulled"
    if not updates:
        return claims, statuses
    data = claims.model_dump(mode="python")
    data.update(updates)
    repaired = ClaimExtraction.model_validate(data)
    validate_evidence_spans(repaired, abstract)
    return repaired, statuses


def _record_span_adjustments(
    run_id: str,
    paper_id: str,
    original_claims: ClaimExtraction,
    final_claims: ClaimExtraction,
    statuses: dict[str, Literal["exact", "repaired", "nulled"]],
) -> None:
    adjusted = [
        field_name
        for field_name, status in statuses.items()
        if status in {"repaired", "nulled"}
    ]
    if not adjusted:
        return
    path = Path(os.getenv("GAPFINDER_RUNS_DIR", str(config.RUNS_DIR))) / run_id
    path.mkdir(parents=True, exist_ok=True)
    with (path / "span_adjustments.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as target:
        for field_name in adjusted:
            original = getattr(original_claims, field_name)
            final = getattr(final_claims, field_name)
            target.write(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "field": field_name,
                        "status": statuses[field_name],
                        "repaired": statuses[field_name] == "repaired",
                        "nulled": statuses[field_name] == "nulled",
                        "original_span": original.evidence_span,
                        "final_span": final.evidence_span,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def recover_cached_batch_responses(
    papers: Sequence[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    cache_dir: Path | None = None,
) -> int:
    """Rebuild cache entries from successful, previously logged raw batch responses."""
    calls_path = Path(os.getenv("GAPFINDER_RUNS_DIR", str(config.RUNS_DIR))) / run_id / "llm_calls.jsonl"
    if not calls_path.is_file():
        return 0
    abstracts = {
        str(paper.get("id") or ""): str(paper.get("abstract") or "").strip()
        for paper in papers
    }
    cache = cache_dir or config.CACHE_DIR / "extraction"
    recovered = 0
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        try:
            call = json.loads(line)
            if call.get("model") != model or call.get("status_code") != 200:
                continue
            payload = json.loads(call.get("response") or "")
            items = payload.get("items")
            if not isinstance(items, list):
                continue
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            paper_id = item.get("paper_id")
            abstract = abstracts.get(str(paper_id))
            if not paper_id or not abstract or not isinstance(item.get("claims"), dict):
                continue
            prompt = _prompt(str(paper_id), abstract)
            target = _cache_path(cache, str(paper_id), model, prompt)
            if target.is_file():
                continue
            try:
                original = ClaimExtraction.model_validate(item["claims"])
                claims, statuses = repair_evidence_spans(original, abstract)
            except (ValidationError, ValueError, TypeError):
                continue
            result = ExtractionResult(
                paper_id=str(paper_id),
                model=model,
                prompt_sha256=_prompt_sha256(prompt),
                claims=claims,
                span_status=statuses,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            _record_span_adjustments(run_id, str(paper_id), original, claims, statuses)
            recovered += 1
    return recovered


def _record_specificity_rejection(
    run_id: str,
    paper_id: str,
    error: ValidationError,
) -> bool:
    specificity_errors = [
        detail
        for detail in error.errors()
        if isinstance(detail.get("ctx", {}).get("error"), SpecificityValidationError)
    ]
    if not specificity_errors:
        return False
    path = Path(os.getenv("GAPFINDER_RUNS_DIR", str(config.RUNS_DIR))) / run_id
    path.mkdir(parents=True, exist_ok=True)
    rejection_path = path / "specificity_rejections.csv"
    exists = rejection_path.is_file()
    with rejection_path.open("a", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=["paper_id", "rejected_fields"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "paper_id": paper_id,
                "rejected_fields": ";".join(
                    str(detail["ctx"]["error"]).split(" ", 1)[0]
                    for detail in specificity_errors
                ),
            }
        )
    return True


def extract_paper(
    paper: dict[str, Any],
    *,
    model: str | None = None,
    cache_dir: Path | None = None,
    run_id: str = "phase3-extraction",
    max_attempts: int = 3,
) -> ExtractionResult:
    """Extract one paper, repairing or nulling individual unsupported spans."""
    paper_id = str(paper.get("id") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    if not paper_id or not abstract:
        raise ValueError("Extraction requires a paper ID and a nonempty abstract.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    model_name = model or config.MODEL
    prompt = _prompt(paper_id, abstract)
    prompt_hash = _prompt_sha256(prompt)
    target_dir = cache_dir or config.CACHE_DIR / "extraction"
    target = _cache_path(target_dir, paper_id, model_name, prompt)
    cached = _read_cache(target, abstract)
    if cached is not None:
        return cached

    if not (
        os.getenv("GEMINI_API_KEY2")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GAPFINDER_API_KEY")
    ):
        raise RuntimeError(
            "Gemini API key is not configured; no extraction request was sent."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    current_prompt = prompt
    for attempt in range(max_attempts):
        try:
            raw = generate(
                current_prompt,
                schema=ClaimExtraction,
                temperature=0.0,
                seed=config.SEED,
                model=model_name,
                run_id=run_id,
                paper_id=paper_id,
                thinking_level="low",
            )
            claims = ClaimExtraction.model_validate_json(raw)
            original_claims = claims
            claims, span_status = repair_evidence_spans(claims, abstract)
            result = ExtractionResult(
                paper_id=paper_id,
                model=model_name,
                prompt_sha256=prompt_hash,
                claims=claims,
                span_status=span_status,
            )
            _record_span_adjustments(
                run_id, paper_id, original_claims, claims, span_status
            )
            target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result
        except ValidationError as exc:
            _record_specificity_rejection(run_id, paper_id, exc)
            last_error = exc
            current_prompt = (
                f"{prompt}\n\nThe prior output failed validation. "
                "Use specific named entities and substantive limitation/future "
                "work, or null when the abstract provides none."
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            current_prompt = (
                f"{prompt}\n\nThe prior output failed validation. "
                "Return valid schema JSON and use exact abstract substrings only."
            )
        except LLMAPIError as exc:
            if exc.status_code == 429:
                raise
            last_error = exc
        except RuntimeError as exc:
            last_error = exc
        if attempt + 1 < max_attempts:
            time.sleep(min(30.0, 2**attempt + random.uniform(0.0, 0.5)))
    raise RuntimeError(
        f"Extraction failed for {paper_id} after {max_attempts} attempts "
        f"({type(last_error).__name__ if last_error else 'unknown error'}): "
        f"{last_error if last_error else 'no error detail'}"
    ) from last_error


def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterator[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def extract_many(
    papers: Iterable[dict[str, Any]],
    *,
    batch_size: int = 10,
    model: str | None = None,
    cache_dir: Path | None = None,
    run_id: str = "phase3-extraction",
) -> list[ExtractionResult]:
    """Process papers in bounded batches; each paper is independently cached/retried."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paper_list = list(papers)
    results: list[ExtractionResult] = []
    for batch in _chunks(paper_list, batch_size):
        results.extend(
            extract_paper(
                paper,
                model=model,
                cache_dir=cache_dir,
                run_id=run_id,
            )
            for paper in batch
        )
    return results


def extract_batch(
    papers: Sequence[dict[str, Any]],
    *,
    model: str,
    thinking_level: str = "low",
    cache_dir: Path | None = None,
    run_id: str = "phase3-extraction",
    request_budget: RequestBudget | None = None,
) -> tuple[list[ExtractionResult], dict[str, str]]:
    """Extract one bounded batch in a single request; return valid rows and failures."""
    if not papers:
        return [], {}
    if len({str(paper.get("id") or "") for paper in papers}) != len(papers):
        raise ValueError("A batch must contain unique paper IDs.")
    cache = cache_dir or config.CACHE_DIR / "extraction"
    pending: list[tuple[dict[str, Any], str, str, Path]] = []
    results: list[ExtractionResult] = []
    failures: dict[str, str] = {}
    for paper in papers:
        paper_id = str(paper.get("id") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not paper_id or not abstract:
            failures[paper_id or "<missing-id>"] = (
                "Extraction requires a paper ID and a nonempty abstract."
            )
            continue
        prompt = _prompt(paper_id, abstract)
        target = _cache_path(cache, paper_id, model, prompt)
        cached = _read_cache(target, abstract)
        if cached is not None:
            results.append(cached)
        else:
            pending.append((paper, paper_id, abstract, target))
    if not pending:
        return results, failures

    instructions = (
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        "Return a JSON object with exactly one `items` array. Each item must contain "
        "one `paper_id` copied exactly from the input and one `claims` object matching "
        "the supplied schema. Include every input paper exactly once; do not add IDs."
    )
    paper_blocks = [
        f"Paper ID: {paper_id}\nTitle: {paper.get('title') or ''}\n"
        f"Year: {paper.get('year') or ''}\nAbstract:\n{abstract}"
        for paper, paper_id, abstract, _target in pending
    ]
    prompt = instructions + "\n\n" + "\n\n---\n\n".join(paper_blocks)
    raw = generate(
        prompt,
        schema=_batch_response_schema(),
        temperature=0.0,
        seed=config.SEED,
        model=model,
        run_id=run_id,
        paper_id=";".join(paper_id for _, paper_id, _, _ in pending),
        thinking_level=thinking_level,
        request_budget=request_budget,
    )
    try:
        payload = json.loads(raw)
        items = payload["items"]
        if not isinstance(items, list):
            raise ValueError("Batch response `items` must be an array.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return results, {
            **failures,
            **{
                paper_id: f"Batch response was not the required JSON object: {exc}"
                for _, paper_id, _, _ in pending
            },
        }

    expected = {paper_id for _, paper_id, _, _ in pending}
    actual_ids = [
        item.get("paper_id") for item in items if isinstance(item, dict)
    ]
    if any(paper_id not in expected for paper_id in actual_ids):
        return results, {
            **failures,
            **{
                paper_id: "Batch response included an unexpected paper ID."
                for _, paper_id, _, _ in pending
            },
        }
    by_id: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str):
            by_id.setdefault(item["paper_id"], []).append(item)
    for paper, paper_id, abstract, target in pending:
        matching = by_id.get(paper_id, [])
        if len(matching) != 1:
            failures[paper_id] = (
                "Batch response did not include this paper exactly once."
            )
            continue
        try:
            claims = ClaimExtraction.model_validate(matching[0]["claims"])
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, ValidationError):
                _record_specificity_rejection(run_id, paper_id, exc)
            failures[paper_id] = f"Batch output failed validation: {exc}"
            continue
        original_claims = claims
        claims, span_status = repair_evidence_spans(claims, abstract)
        _record_span_adjustments(run_id, paper_id, original_claims, claims, span_status)
        result = ExtractionResult(
            paper_id=paper_id,
            model=model,
            prompt_sha256=_prompt_sha256(_prompt(paper_id, abstract)),
            claims=claims,
            span_status=span_status,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        results.append(result)
    return results, failures
