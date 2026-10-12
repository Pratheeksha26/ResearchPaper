from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from gapfinder.extraction.extractor import ExtractionResult

FIELDS = (
    "task",
    "method",
    "dataset",
    "metric",
    "setting",
    "stated_limitation",
    "stated_future_work",
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def score_extractions(
    annotations: list[dict[str, str]],
    predictions: list[ExtractionResult],
) -> dict[str, dict[str, float | int | None]]:
    """Compute exact normalized field-value precision/recall/F1 against filled labels."""
    predicted_by_id = {item.paper_id: item.claims for item in predictions}
    counts: dict[str, Counter[str]] = {
        field: Counter() for field in FIELDS
    }
    for row in annotations:
        paper_id = row["paper_id"]
        prediction = predicted_by_id.get(paper_id)
        for field in FIELDS:
            gold = _normalize(row.get(f"{field}_label", ""))
            if not gold:
                continue
            predicted_field = getattr(prediction, field) if prediction else None
            predicted = (
                _normalize(predicted_field.value)
                if predicted_field and predicted_field.value
                else ""
            )
            if predicted == gold:
                counts[field]["tp"] += 1
            else:
                counts[field]["fn"] += 1
                if predicted:
                    counts[field]["fp"] += 1
    scores: dict[str, dict[str, float | int | None]] = {}
    for field, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        scores[field] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return scores


def _load_predictions(path: Path) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                results.append(ExtractionResult.model_validate_json(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score extracted fields against a manually completed annotation CSV."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.annotations.open(encoding="utf-8-sig", newline="") as source:
        annotations = list(csv.DictReader(source))
    scores = score_extractions(annotations, _load_predictions(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
