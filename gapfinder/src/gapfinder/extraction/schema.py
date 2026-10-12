"""Structured claim schema and heuristic specificity validation.

The specificity stoplist has known gaps for moderately generic method phrasing
and may need a broader review; current checks target exact generic phrases and
counted aggregate labels rather than fully judging semantic specificity.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, model_validator

PaperType = Literal[
    "survey_or_review",
    "new_method",
    "benchmark_or_dataset",
    "empirical_study",
    "application",
    "position_or_other",
]

GENERIC_CLAIM_PHRASES = frozenset(
    {
        "benchmark data set",
        "benchmark data sets",
        "benchmark dataset",
        "benchmark datasets",
        "various dataset",
        "various datasets",
        "different datasets",
        "multiple datasets",
        "novel approach",
        "new approach",
        "adaptive model",
        "proposed approach",
        "our approach",
        "four open problems",
        "four open problems for future research",
        "potential research directions",
        "future work",
        "future research",
        "several limitations",
        "limitations",
        "more research is needed",
        "further work",
    }
)

_COUNT_PREFIX = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|several|multiple|various|numerous)"
)
_GENERIC_COUNT_DESCRIPTORS = (
    r"(?:(?:public|real|world|synthetic|benchmark|graph|neural|convolutional|"
    r"attention|message|passing|network|standard|common|popular|established|"
    r"challenging|large|scale|small|diverse|different|classification|prediction|"
    r"learning|training|open|source)\s+)*"
)
_COUNTED_GENERIC_CLAIMS = {
    "dataset": re.compile(
        rf"^{_COUNT_PREFIX}\s+{_GENERIC_COUNT_DESCRIPTORS}"
        r"(?:data\s+sets?|datasets?|benchmarks?)\b"
    ),
    "task": re.compile(
        rf"^{_COUNT_PREFIX}\s+{_GENERIC_COUNT_DESCRIPTORS}"
        r"(?:tasks?|problems?|objectives?|goals?|challenges?)\b"
    ),
    "method": re.compile(
        rf"^{_COUNT_PREFIX}\s+{_GENERIC_COUNT_DESCRIPTORS}"
        r"(?:methods?|models?|algorithms?|approaches?|architectures?|networks?|variants?)\b"
    ),
}


class SpecificityValidationError(ValueError):
    """Model output used a generic phrase where a named claim is required."""


def _normalized_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def is_generic_claim_value(field_name: str, value: str) -> bool:
    """Return whether a task/method/dataset claim matches the generic stoplist."""
    normalized = _normalized_phrase(value)
    if normalized in GENERIC_CLAIM_PHRASES:
        return True
    pattern = _COUNTED_GENERIC_CLAIMS.get(field_name)
    return pattern is not None and pattern.match(normalized) is not None


class EvidenceValue(BaseModel):
    """A field value paired with its verbatim abstract evidence, when stated."""

    value: str | None
    evidence_span: str | None

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) != {"value", "evidence_span"}:
            raise ValueError("EvidenceValue contains unknown or missing fields")
        return value

    @model_validator(mode="after")
    def require_value_and_span_together(self) -> EvidenceValue:
        if (self.value is None) != (self.evidence_span is None):
            raise ValueError("value and evidence_span must both be present or both null")
        return self


class ClaimExtraction(BaseModel):
    """Paper-level claims grounded to exact substrings of the abstract."""

    paper_type: PaperType
    include_in_counts: bool
    task: EvidenceValue
    method: EvidenceValue
    dataset: EvidenceValue
    metric: EvidenceValue
    setting: EvidenceValue
    stated_limitation: EvidenceValue
    stated_future_work: EvidenceValue

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) != set(cls.model_fields):
            raise ValueError("ClaimExtraction contains unknown or missing fields")
        return value

    @model_validator(mode="after")
    def validate_claim_specificity_and_paper_type(self) -> ClaimExtraction:
        if self.paper_type == "survey_or_review":
            if any(
                getattr(self, name).value is not None
                for name in ("task", "method", "dataset")
            ):
                raise ValueError(
                    "Survey/review task, method, and dataset must all be null"
                )
            if self.include_in_counts:
                raise ValueError(
                    "Survey/review papers must set include_in_counts=false"
                )

        for field_name in ("task", "method", "dataset"):
            value = getattr(self, field_name).value
            if value and is_generic_claim_value(field_name, value):
                raise SpecificityValidationError(
                    f"{field_name} is too generic to identify a named entity"
                )

        for field_name in ("stated_limitation", "stated_future_work"):
            value = getattr(self, field_name).value
            if value and _normalized_phrase(value) in GENERIC_CLAIM_PHRASES:
                raise SpecificityValidationError(
                    f"{field_name} is only a generic pointer, not substantive content"
                )
        return self


def validate_evidence_spans(extraction: ClaimExtraction, abstract: str) -> None:
    """Reject any nonempty evidence span that is not an exact abstract substring."""
    for field_name in (
        "task",
        "method",
        "dataset",
        "metric",
        "setting",
        "stated_limitation",
        "stated_future_work",
    ):
        evidence = getattr(extraction, field_name)
        if evidence.evidence_span is not None and evidence.evidence_span not in abstract:
            raise ValueError(
                f"{field_name} evidence_span is not an exact substring of the abstract"
            )
