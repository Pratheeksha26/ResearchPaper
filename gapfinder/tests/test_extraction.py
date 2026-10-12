from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from gapfinder.eval.extraction_eval import score_extractions
from gapfinder.extraction import extractor
from gapfinder.extraction.annotation_sheet import create_annotation_sheet
from gapfinder.extraction.extractor import (
    ExtractionResult,
    extract_batch,
    extract_paper,
    recover_cached_batch_responses,
    repair_evidence_spans,
)
from gapfinder.extraction.extractor import _batch_response_schema, _prompt
from gapfinder.extraction.normalize import cluster_entities, normalize_claim_entities
from gapfinder.extraction.run import (
    _append_batch_attempt,
    _classify_span,
    _read_papers,
)
from gapfinder.extraction.schema import (
    ClaimExtraction,
    EvidenceValue,
    is_generic_claim_value,
    validate_evidence_spans,
)
from gapfinder.llm.rate_limiter import ModelLimits


def _claims(span: str = "We evaluate on Cora.") -> ClaimExtraction:
    empty = EvidenceValue(value=None, evidence_span=None)
    return ClaimExtraction(
        paper_type="new_method",
        include_in_counts=True,
        task=empty,
        method=EvidenceValue(value="graph neural network", evidence_span=span),
        dataset=empty,
        metric=empty,
        setting=empty,
        stated_limitation=empty,
        stated_future_work=empty,
    )


def test_schema_requires_exact_substrings_and_nulls_for_absent_fields() -> None:
    claims = _claims()
    validate_evidence_spans(claims, "We evaluate on Cora.")
    with pytest.raises(ValueError, match="exact substring"):
        validate_evidence_spans(claims, "Evaluation used Cora.")
    with pytest.raises(ValueError, match="both be present"):
        EvidenceValue(value="graph", evidence_span=None)


def test_exact_span_validator_rejects_deliberately_scrambled_span() -> None:
    claims = _claims("predicting academic this paper")
    abstract = (
        "We present a novel model for predicting academic performance in this paper."
    )
    with pytest.raises(ValueError, match="exact substring"):
        validate_evidence_spans(claims, abstract)


def test_span_repair_uses_literal_substrings_and_nulls_unmatched_fields() -> None:
    data = _claims("A graph neural network for node classification").model_dump(
        mode="json"
    )
    data["task"] = {
        "value": "astronomy graph forecasting",
        "evidence_span": "forecasting astronomy graph for the galaxy",
    }
    claims = ClaimExtraction.model_validate(data)
    abstract = (
        "We propose a graph neural network for node classification. "
        "The approach evaluates accuracy on Cora."
    )
    repaired, statuses = repair_evidence_spans(claims, abstract)
    assert repaired.method.evidence_span == (
        "a graph neural network for node classification"
    )
    assert "a graph neural network for node classification" in abstract
    assert repaired.task.value is None
    assert repaired.task.evidence_span is None
    assert statuses == {"task": "nulled", "method": "repaired"}
    validate_evidence_spans(repaired, abstract)


def test_span_audit_distinguishes_exact_near_miss_and_fabricated() -> None:
    abstract = "Graph neural networks model molecular property prediction."
    assert _classify_span("Graph neural networks", abstract) == "exact"
    assert _classify_span("neural GRAPH networks", abstract) == "near_miss"
    assert _classify_span("quadratic enzyme chemistry reactions", abstract) == "fabricated"


def test_extraction_prompt_requires_character_exact_evidence_span() -> None:
    prompt = _prompt("W123", "We evaluate the model on Cora.")
    assert "unmodified substring" in prompt
    assert "character-for-character" in prompt
    assert 'Given text "We evaluate the model on Cora."' in prompt
    assert "the correct evidence_span is" in prompt
    assert '"evaluate the model on Cora"' in prompt


def test_extractor_caches_validated_results_without_second_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key")
    paper = {"id": "W123", "abstract": "We evaluate on Cora."}
    calls = []
    response = _claims().model_dump_json()

    def fake_generate(*_args, **_kwargs):
        calls.append(1)
        return response

    monkeypatch.setattr(extractor, "generate", fake_generate)
    first = extract_paper(paper, model="test-model", cache_dir=tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY")
    second = extract_paper(paper, model="test-model", cache_dir=tmp_path)
    assert first == second
    assert len(calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_extract_batch_sends_one_request_and_caches_each_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAPFINDER_SEED", "42")
    papers = [
        {"id": f"W{index}", "title": f"Paper {index}", "abstract": "We evaluate on Cora."}
        for index in range(3)
    ]
    calls: list[dict[str, object]] = []

    def fake_generate(prompt, **kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "items": [
                    {
                        "paper_id": paper["id"],
                        "claims": _claims().model_dump(mode="json"),
                    }
                    for paper in papers
                ]
            }
        )

    monkeypatch.setattr("gapfinder.extraction.extractor.generate", fake_generate)
    results, failures = extract_batch(
        papers,
        model="gemini-3.1-flash-lite",
        cache_dir=tmp_path,
    )
    assert len(results) == 3
    assert failures == {}
    assert len(calls) == 1
    assert calls[0]["thinking_level"] == "low"
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_extract_batch_passes_thinking_level_to_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = {"id": "W-minimal", "abstract": "We evaluate on Cora."}
    received: list[str] = []

    def fake_generate(_prompt, **kwargs):
        received.append(kwargs["thinking_level"])
        return json.dumps(
            {
                "items": [
                    {
                        "paper_id": paper["id"],
                        "claims": _claims().model_dump(mode="json"),
                    }
                ]
            }
        )

    monkeypatch.setattr("gapfinder.extraction.extractor.generate", fake_generate)
    results, failures = extract_batch(
        [paper],
        model="gemini-3.1-flash-lite",
        thinking_level="minimal",
        cache_dir=tmp_path,
    )

    assert [result.paper_id for result in results] == ["W-minimal"]
    assert failures == {}
    assert received == ["minimal"]


def test_batch_json_schema_hoists_nested_definitions() -> None:
    schema = _batch_response_schema()
    item_schema = schema["properties"]["items"]["items"]
    claims_schema = item_schema["properties"]["claims"]
    assert "$defs" in schema
    assert "$defs" not in claims_schema
    assert claims_schema["properties"]["task"]["$ref"].startswith("#/$defs/")
    json.loads(json.dumps(schema))

    def resolve_reference(document: dict, reference: str) -> object:
        assert reference.startswith("#/")
        target: object = document
        for part in reference[2:].split("/"):
            assert isinstance(target, dict)
            target = target[part]
        return target

    def references(value: object):
        if isinstance(value, dict):
            if "$ref" in value:
                yield value["$ref"]
            for nested in value.values():
                yield from references(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from references(nested)

    assert all(resolve_reference(schema, ref) for ref in references(schema))


def test_nested_batch_definitions_reproduce_the_broken_schema_layout() -> None:
    claim_schema = ClaimExtraction.model_json_schema()
    malformed_legacy_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"claims": claim_schema},
                },
            }
        },
    }
    nested_ref = claim_schema["properties"]["task"]["$ref"]
    assert nested_ref.startswith("#/$defs/")
    assert "$defs" not in malformed_legacy_schema
    assert "$defs" in malformed_legacy_schema["properties"]["items"]["items"]["properties"]["claims"]
    with pytest.raises(KeyError):
        target: object = malformed_legacy_schema
        for part in nested_ref[2:].split("/"):
            assert isinstance(target, dict)
            target = target[part]


def test_extract_batch_reports_missing_papers_for_single_item_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = [
        {"id": "W1", "abstract": "We evaluate on Cora."},
        {"id": "W2", "abstract": "We evaluate on Cora."},
    ]

    def fake_generate(_prompt, **_kwargs):
        return json.dumps(
            {
                "items": [
                    {"paper_id": "W1", "claims": _claims().model_dump(mode="json")}
                ]
            }
        )

    monkeypatch.setattr("gapfinder.extraction.extractor.generate", fake_generate)
    results, failures = extract_batch(
        papers, model="gemini-3.1-flash-lite", cache_dir=tmp_path
    )
    assert [result.paper_id for result in results] == ["W1"]
    assert failures == {"W2": "Batch response did not include this paper exactly once."}


def test_extract_batch_repairs_invalid_span_without_rejecting_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = {
        "id": "W-span-repair",
        "abstract": "We propose a graph neural network for node classification.",
    }
    claims = _claims("graph neural network for classification")

    def fake_generate(_prompt, **_kwargs):
        return json.dumps(
            {
                "items": [
                    {"paper_id": paper["id"], "claims": claims.model_dump(mode="json")}
                ]
            }
        )

    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("gapfinder.extraction.extractor.generate", fake_generate)
    results, failures = extract_batch(
        [paper],
        model="gemini-3.1-flash-lite",
        cache_dir=tmp_path / "cache",
        run_id="repair-test",
    )
    assert failures == {}
    assert len(results) == 1
    assert results[0].span_status == {"method": "repaired"}
    assert results[0].claims.method.evidence_span == (
        "graph neural network for node classification"
    )
    adjustment = json.loads(
        (tmp_path / "runs" / "repair-test" / "span_adjustments.jsonl")
        .read_text(encoding="utf-8")
    )
    assert adjustment["repaired"] is True
    assert adjustment["nulled"] is False


def test_recover_cached_batch_responses_repairs_without_api_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = {
        "id": "W-recover",
        "abstract": "We propose a graph neural network for node classification.",
    }
    original = _claims("A graph neural network for node classification")
    run_id = "recover-test"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "llm_calls.jsonl").write_text(
        json.dumps(
            {
                "model": "gemini-3.1-flash-lite",
                "status_code": 200,
                "response": json.dumps(
                    {
                        "items": [
                            {
                                "paper_id": paper["id"],
                                "claims": original.model_dump(mode="json"),
                            }
                        ]
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.1-flash-lite")
    recovered = recover_cached_batch_responses(
        [paper],
        model="gemini-3.1-flash-lite",
        run_id=run_id,
        cache_dir=tmp_path / "cache",
    )
    assert recovered == 1
    prompt = extractor._prompt(paper["id"], paper["abstract"])
    target = extractor._cache_path(
        tmp_path / "cache", paper["id"], "gemini-3.1-flash-lite", prompt
    )
    cached = ExtractionResult.model_validate_json(target.read_text(encoding="utf-8"))
    assert cached.span_status == {"method": "repaired"}
    assert cached.claims.method.evidence_span == (
        "a graph neural network for node classification"
    )


def test_batch_attempt_journal_preserves_request_and_validation_outcomes(
    tmp_path: Path,
) -> None:
    _append_batch_attempt(
        tmp_path,
        {
            "batch_number": 1,
            "phase": "primary",
            "paper_ids": ["W1", "W2"],
            "successful_ids": ["W1"],
            "failures": {"W2": "evidence span is not exact"},
            "request_attempts": 1,
        },
    )
    row = json.loads((tmp_path / "batch_attempts.jsonl").read_text(encoding="utf-8"))
    assert row["paper_ids"] == ["W1", "W2"]
    assert row["successful_ids"] == ["W1"]
    assert row["failures"] == {"W2": "evidence span is not exact"}
    assert row["request_attempts"] == 1


def test_extractor_fails_before_retrying_when_api_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY2", "")
    monkeypatch.delenv("GAPFINDER_API_KEY", raising=False)

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("generate must not be called when credentials are absent")

    monkeypatch.setattr(extractor, "generate", unexpected_call)
    with pytest.raises(RuntimeError, match="no extraction request was sent"):
        extract_paper(
            {"id": "W123", "abstract": "Abstract."},
            cache_dir=tmp_path,
        )


def test_entity_normalization_clusters_cosine_neighbors() -> None:
    mapping = cluster_entities(
        ["GNN", "graph neural network", "dataset"],
        np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
        similarity_threshold=0.9,
    )
    assert mapping["GNN"] == mapping["graph neural network"]
    assert mapping["dataset"] == "dataset"


def test_complete_linkage_does_not_join_weak_transitive_endpoint() -> None:
    angles = np.deg2rad([0.0, 20.0, 40.0])
    embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
    mapping = cluster_entities(
        ["A", "B", "C"], embeddings, similarity_threshold=0.9
    )

    assert mapping["A"] != mapping["C"]
    assert len(set(mapping.values())) == 2


def test_normalization_never_clusters_across_field_types() -> None:
    claims = _claims()
    data = claims.model_dump()
    data["task"] = {"value": "same label", "evidence_span": "same label"}
    data["method"] = {"value": "same label", "evidence_span": "same label"}
    data["dataset"] = {"value": "same label", "evidence_span": "same label"}
    claims = ClaimExtraction.model_validate(data)

    class IdenticalEncoder:
        @staticmethod
        def encode(sentences, **_kwargs):
            return np.ones((len(sentences), 2), dtype=np.float32)

    mappings = normalize_claim_entities([claims], IdenticalEncoder())

    assert set(mappings) == {"task", "method", "dataset", "metric", "setting"}
    assert mappings["task"] == {"same label": "same label"}
    assert mappings["method"] == {"same label": "same label"}
    assert mappings["dataset"] == {"same label": "same label"}


def test_annotation_sheet_has_blank_label_columns(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {"id": "W1", "title": "Paper one", "year": 2024, "abstract": "Abstract."}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "annotation.csv"
    assert create_annotation_sheet(corpus, output, sample_size=1) == 1
    with output.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert all(not row[name] for name in row if name.endswith("_label"))
    assert row["abstract"] == "Abstract."


def test_comparison_requires_the_exact_100_paper_annotation_sample(
    tmp_path: Path,
) -> None:
    from gapfinder.extraction.run import run_extraction

    sample = tmp_path / "extraction_sample.csv"
    sample.write_text("paper_id,title,year,abstract\nW1,Paper,2024,Abstract.\n")
    with pytest.raises(ValueError, match="exactly 100"):
        run_extraction(
            corpus_path=sample,
            output_dir=tmp_path / "run",
            comparison_approved=True,
        )


def test_rate_limiter_pauses_until_the_next_pacific_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gapfinder.llm.rate_limiter as limiter

    model = "gemini-3.1-flash-lite"
    monkeypatch.setenv("GAPFINDER_MODEL", model)
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setitem(limiter.MODEL_LIMITS, model, ModelLimits(15, 250_000, 1))
    days = iter(("2026-09-28", "2026-09-28", "2026-09-29"))
    monkeypatch.setattr(limiter, "_pacific_date", lambda: next(days))
    monkeypatch.setattr(limiter, "_seconds_until_pacific_reset", lambda: 1.0)
    sleeps: list[float] = []
    pause_callbacks: list[str] = []
    monkeypatch.setattr(limiter.time, "sleep", sleeps.append)

    limiter.reserve_request(model, 10)
    limiter.reserve_request(
        model, 10, on_daily_pause=lambda: pause_callbacks.append("paused")
    )

    assert sleeps == [1.0]
    assert pause_callbacks == ["paused"]
    state = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert state["pacific_date"] == "2026-09-29"
    assert state["models"][model]["attempts_today"] == 1


def test_survey_schema_requires_null_core_claims_and_excludes_counts() -> None:
    empty = EvidenceValue(value=None, evidence_span=None)
    with pytest.raises(ValueError, match="must all be null"):
        ClaimExtraction(
            paper_type="survey_or_review",
            include_in_counts=False,
            task=EvidenceValue(value="graph classification", evidence_span="classification"),
            method=empty,
            dataset=empty,
            metric=empty,
            setting=empty,
            stated_limitation=empty,
            stated_future_work=empty,
        )
    with pytest.raises(ValueError, match="include_in_counts=false"):
        ClaimExtraction(
            paper_type="survey_or_review",
            include_in_counts=True,
            task=empty,
            method=empty,
            dataset=empty,
            metric=empty,
            setting=empty,
            stated_limitation=empty,
            stated_future_work=empty,
        )


@pytest.mark.parametrize(
    ("field", "phrase"),
    [
        ("task", "various datasets"),
        ("method", "novel approach"),
        ("dataset", "benchmark data sets"),
        ("stated_future_work", "four open problems for future research"),
        ("stated_limitation", "several limitations"),
    ],
)
def test_specificity_validator_rejects_generic_claims(
    field: str, phrase: str
) -> None:
    data = _claims().model_dump()
    data[field] = {"value": phrase, "evidence_span": phrase}
    with pytest.raises(ValueError, match="too generic|generic pointer"):
        ClaimExtraction.model_validate(data)


@pytest.mark.parametrize(
    ("field", "phrase"),
    [
        ("dataset", "14 real-world graph datasets"),
        ("dataset", "10 benchmark datasets"),
        ("dataset", "three public datasets"),
        ("task", "several graph classification tasks"),
        ("method", "two graph neural network models"),
    ],
)
def test_specificity_validator_rejects_counted_generic_claims(
    field: str, phrase: str
) -> None:
    data = _claims().model_dump()
    data[field] = {"value": phrase, "evidence_span": phrase}
    with pytest.raises(ValueError, match="too generic"):
        ClaimExtraction.model_validate(data)


@pytest.mark.parametrize(
    ("field", "phrase"),
    [
        ("dataset", "3 UCI repositories"),
        ("dataset", "3D molecular graphs"),
        ("task", "two-stage node classification"),
        ("method", "3D graph neural network architecture"),
    ],
)
def test_specificity_validator_preserves_counted_specific_labels(
    field: str, phrase: str
) -> None:
    assert not is_generic_claim_value(field, phrase)


@pytest.mark.parametrize(
    "configured_model",
    ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemma-4-31b-it"],
)
def test_specificity_rejection_is_independent_of_selected_model(
    configured_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAPFINDER_MODEL", configured_model)
    data = _claims().model_dump()
    data["method"] = {
        "value": "Adaptive model",
        "evidence_span": "Adaptive model",
    }
    with pytest.raises(ValidationError):
        ClaimExtraction.model_validate(data)


def test_specificity_validator_rejections_are_countable_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    data = _claims().model_dump()
    data["dataset"] = {
        "value": "benchmark data sets",
        "evidence_span": "benchmark data sets",
    }
    with pytest.raises(ValidationError) as error:
        ClaimExtraction.model_validate(data)
    assert extractor._record_specificity_rejection(
        "specificity-test", "W123", error.value
    )
    with (tmp_path / "specificity-test" / "specificity_rejections.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    assert rows == [{"paper_id": "W123", "rejected_fields": "dataset"}]


def test_annotation_sample_is_proportionally_stratified_by_year(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": f"W{year}-{index}",
            "year": year,
            "title": f"Paper {year}-{index}",
            "abstract": "Abstract.",
        }
        for year, amount in ((2020, 9), (2021, 1))
        for index in range(amount)
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    create_annotation_sheet(corpus, first, sample_size=5, seed=101)
    create_annotation_sheet(corpus, second, sample_size=5, seed=101)
    assert first.read_bytes() == second.read_bytes()
    with first.open(encoding="utf-8", newline="") as source:
        years = [row["year"] for row in csv.DictReader(source)]
    assert years.count("2020") == 4
    assert years.count("2021") == 1


def test_100_paper_sample_guarantees_five_per_target_year_and_blank_labels(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": f"W{year}-{index}",
            "year": year,
            "title": f"Paper {year}-{index}",
            "abstract": "Abstract.",
        }
        for year in range(2015, 2027)
        for index in range(20)
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "sample.csv"
    assert create_annotation_sheet(
        corpus, output, sample_size=100, seed=20260928
    ) == 100
    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    counts = {
        year: sum(row["year"] == str(year) for row in rows)
        for year in range(2019, 2026)
    }
    assert all(count >= 5 for count in counts.values())
    assert all(
        not row[field]
        for row in rows
        for field in row
        if field.endswith("_label")
    )


def test_extraction_input_rejects_synthetic_papers(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "fixture:1",
                "title": "Synthetic",
                "abstract": "Fixture text.",
                "synthetic": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Synthetic"):
        _read_papers(corpus)


def test_extraction_eval_scores_only_filled_manual_labels() -> None:
    prediction = ExtractionResult(
        paper_id="W1",
        model="test-model",
        prompt_sha256="hash",
        claims=_claims(),
    )
    scores = score_extractions(
        [{"paper_id": "W1", "method_label": "Graph Neural Network"}],
        [prediction],
    )
    assert scores["method"]["tp"] == 1
    assert scores["method"]["f1"] == 1.0
    assert scores["task"]["tp"] == 0
