from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _is_english_text(text: str | None) -> bool:
    if text is None:
        return False
    cleaned = text.strip()
    if not cleaned:
        return False
    ascii_letters = sum(1 for ch in cleaned if ch.isascii() and ch.isalpha())
    return ascii_letters > 0


def _passes_filters(record: dict[str, Any], *, min_year: int = 2015) -> bool:
    abstract = str(record.get("abstract") or "")
    title = str(record.get("title") or "")
    year = int(record.get("year") or 0)
    if not abstract.strip():
        return False
    if len(abstract.split()) < 5:
        return False
    if year < min_year:
        return False
    if not _is_english_text(title) or not _is_english_text(abstract):
        return False
    return True


def screen_records(
    records: list[dict[str, Any]], *, min_year: int = 2015
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Separate included records from exclusions while preserving each exclusion reason."""
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts = {"abstract_missing": 0, "language_date_filtered": 0}
    for record in records:
        abstract = str(record.get("abstract") or "")
        if not abstract.strip():
            excluded.append({**record, "exclusion_reason": "abstract_missing"})
            counts["abstract_missing"] += 1
        elif not _passes_filters(record, min_year=min_year):
            reasons = []
            if len(abstract.split()) < 5:
                reasons.append("abstract_too_short")
            if int(record.get("year") or 0) < min_year:
                reasons.append("before_minimum_year")
            if not _is_english_text(str(record.get("title") or "")) or not _is_english_text(abstract):
                reasons.append("non_english")
            excluded.append({**record, "exclusion_reason": ",".join(reasons)})
            counts["language_date_filtered"] += 1
        else:
            included.append(record)
    return included, excluded, counts


def make_prisma_counts(records: list[dict[str, Any]], *, min_year: int = 2015) -> dict[str, Any]:
    """Create a deterministic PRISMA-style count snapshot for a corpus."""
    identified = len(records)
    deduped = len(records)
    included_records, _excluded, screening_counts = screen_records(records, min_year=min_year)
    abstract_missing_removed = screening_counts["abstract_missing"]
    language_date_filtered = screening_counts["language_date_filtered"]
    after_abstract = identified - abstract_missing_removed

    return {
        "identified": identified,
        "deduped": deduped,
        "abstract_missing_removed": abstract_missing_removed,
        "after_abstract_missing_removal": after_abstract,
        "language_date_filtered": language_date_filtered,
        "after_language_date_filtering": len(included_records),
        "included": len(included_records),
    }


def export_prisma(prisma: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(prisma, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def export_flow_diagram(prisma: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "graph TD",
        f"A[Identified: {prisma.get('identified', 0)}] --> B[After DOI dedupe: {prisma.get('after_doi_dedupe', prisma.get('deduped', 0))}]",
        f"B --> C[After title dedupe: {prisma.get('after_title_dedupe', prisma.get('deduped', 0))}]",
        f"C --> D[After embedding dedupe: {prisma.get('after_embedding_dedupe', prisma.get('deduped', 0))}]",
        f"D --> E[Abstract missing removed: {prisma.get('abstract_missing_removed', 0)}]",
        f"E --> F[After abstract removal: {prisma.get('after_abstract_missing_removal', 0)}]",
        f"F --> G[Language/date filtered: {prisma.get('language_date_filtered', 0)}]",
    ]
    cap_removed = int(prisma.get("yearly_cap_removed", 0))
    if cap_removed:
        lines.extend(
            [
                f"G --> H[Eligible after screening: {prisma.get('after_language_date_filtering', 0)}]",
                f"H --> I[Yearly cap removed: {cap_removed}]",
                f"I --> J[Included: {prisma.get('included', 0)}]",
            ]
        )
    else:
        lines.append(f"G --> H[Included: {prisma.get('included', 0)}]")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = ["make_prisma_counts", "export_prisma", "export_flow_diagram"]
