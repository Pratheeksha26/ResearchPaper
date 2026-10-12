from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from gapfinder.config import OPENALEX_API_KEY
from gapfinder.data.dedupe import dedupe_records
from gapfinder.data.openalex import (
    export_records,
    get_rate_limit,
    search_openalex_phrase_with_metadata,
)
from gapfinder.data.prisma import export_flow_diagram, export_prisma, screen_records

YEARS = tuple(range(2015, 2027))
SORT_ORDER = "relevance_score:desc"
MAX_INCLUDED_PER_YEAR = 200


def _budget_summary(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "daily_budget",
        "daily_remaining",
        "monthly_budget",
        "monthly_remaining",
        "requests_per_second_limit",
        "rate_limit",
        "remaining",
        "limit",
    }
    return {key: payload[key] for key in allowed if key in payload}


def _count_by_year(records: list[dict[str, Any]]) -> Counter[int]:
    return Counter(int(record["year"]) for record in records if record.get("year"))


def _cap_records_per_year(
    records: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], Counter[int]]:
    included: list[dict[str, Any]] = []
    kept_by_year: Counter[int] = Counter()
    removed_by_year: Counter[int] = Counter()
    for record in records:
        year = int(record.get("year") or 0)
        if kept_by_year[year] >= limit:
            removed_by_year[year] += 1
            continue
        included.append(record)
        kept_by_year[year] += 1
    return included, removed_by_year


def build_year_stratified_corpus(
    topic: str,
    *,
    max_per_year: int = 300,
    output_dir: str | Path = ".",
    corpus_name: str = "corpus_main_v4.jsonl",
    prisma_name: str = "prisma_main_v4.json",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= max_per_year <= 2000:
        raise ValueError("max_per_year must be between 1 and 2000.")

    print(
        "OpenAlex rate limit at build start: "
        + json.dumps(_budget_summary(get_rate_limit()), sort_keys=True)
    )
    raw_records: list[dict[str, Any]] = []
    per_year: dict[str, dict[str, Any]] = {}
    try:
        for year in YEARS:
            records, metadata = search_openalex_phrase_with_metadata(
                publication_year=year,
                max_results=max_per_year,
                per_page=100,
            )
            raw_records.extend(records)
            per_year[str(year)] = {
                "identified": len(records),
                "total_matches": metadata["total_matches"],
                "sort_order": metadata["sort"],
                "filter": metadata["filter"],
                "search_parameter": metadata["search_parameter"],
                "search_phrases": metadata["search_phrases"],
                "title_abstract_phrase_required": metadata[
                    "title_abstract_phrase_required"
                ],
                "api_results_scanned": metadata["api_results_scanned"],
                "pages_fetched": metadata["pages_fetched"],
                "language_filter_applied": (
                    metadata["language_filter"] == "language:en"
                ),
            }

        dedupe_result = dedupe_records(raw_records, return_stats=True)
        if not isinstance(dedupe_result, tuple):
            raise TypeError("Deduplication did not return its requested stage counts.")
        unique, dedupe_counts = dedupe_result
        after_doi = len(raw_records) - dedupe_counts["doi_exact"]
        after_title = after_doi - dedupe_counts["title_normalized"]
        if len(unique) != after_title - dedupe_counts["embedding_cosine"]:
            raise AssertionError("Deduplication stage counts do not reconcile.")
        eligible_records, excluded, screening_counts = screen_records(
            unique, min_year=min(YEARS)
        )
        included, yearly_cap_removed = _cap_records_per_year(
            eligible_records, MAX_INCLUDED_PER_YEAR
        )

        included_by_year = _count_by_year(included)
        eligible_by_year = _count_by_year(eligible_records)

        included_by_year_counter = Counter(included_by_year)
        for year in YEARS:
            key = str(year)
            per_year[key].update(
                {
                    "included": included_by_year_counter[year],
                    "eligible_after_screening": eligible_by_year[year],
                    "yearly_cap_removed": yearly_cap_removed[year],
                }
            )

        prisma: dict[str, Any] = {
            "topic": topic,
            "year_range": [min(YEARS), max(YEARS)],
            "max_results_per_year": max_per_year,
            "max_included_per_year": MAX_INCLUDED_PER_YEAR,
            "per_page": 100,
            "identified": len(raw_records),
            "after_doi_dedupe": after_doi,
            "after_title_dedupe": after_title,
            "after_embedding_dedupe": len(unique),
            "embedding_dedupe_status": (
                "applied"
                if any(isinstance(row.get("embedding"), list) for row in unique)
                else "skipped"
            ),
            "doi_duplicates_removed": dedupe_counts["doi_exact"],
            "title_duplicates_removed": dedupe_counts["title_normalized"],
            "embedding_duplicates_removed": dedupe_counts["embedding_cosine"],
            "abstract_missing_removed": screening_counts["abstract_missing"],
            "after_abstract_missing_removal": (
                len(unique) - screening_counts["abstract_missing"]
            ),
            "language_date_filtered": screening_counts[
                "language_date_filtered"
            ],
            "after_language_date_filtering": len(eligible_records),
            "yearly_cap_removed": sum(yearly_cap_removed.values()),
            "after_yearly_included_cap": len(included),
            "included": len(included),
            "language_filter": "language:en",
            "language_filter_applied": True,
            "sort_order": SORT_ORDER,
            "per_year": per_year,
            "openalex_total_matches_sum_by_year": sum(
                int(info["total_matches"] or 0) for info in per_year.values()
            ),
        }
        if any(record.get("synthetic") is True for record in unique):
            raise ValueError("Synthetic records cannot be written as the main corpus.")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if len(included) != prisma["included"]:
            raise AssertionError(
                "Included corpus length does not match the PRISMA included count."
            )
        export_records(included, output / corpus_name)
        export_records(
            excluded,
            output / f"excluded_{Path(corpus_name).stem}.jsonl",
        )
        export_prisma(prisma, output / prisma_name)
        export_flow_diagram(prisma, output / f"{Path(prisma_name).stem}.md")
        return included, prisma
    except requests.RequestException as exc:
        status_code = exc.response.status_code if exc.response is not None else "unavailable"
        print(f"OpenAlex API error: HTTP {status_code}", file=sys.stderr)
        if exc.response is not None:
            body = exc.response.text
            if OPENALEX_API_KEY:
                body = body.replace(OPENALEX_API_KEY, "[REDACTED]")
            print(body, file=sys.stderr)
        raise
    finally:
        try:
            end_budget = _budget_summary(get_rate_limit())
        except requests.RequestException as exc:
            status_code = (
                exc.response.status_code if exc.response is not None else "unavailable"
            )
            print(
                f"OpenAlex rate limit at build end: unavailable (HTTP {status_code})",
                file=sys.stderr,
            )
        else:
            print(
                "OpenAlex rate limit at build end: "
                + json.dumps(end_budget, sort_keys=True)
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a year-stratified OpenAlex corpus."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument(
        "--max-per-year",
        type=int,
        default=300,
        help="Maximum strict title/abstract phrase matches to scan per year before screening.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--corpus-name", default="corpus_main_v4.jsonl")
    parser.add_argument("--prisma-name", default="prisma_main_v4.json")
    args = parser.parse_args()
    if not 1 <= args.max_per_year <= 2000:
        parser.error("--max-per-year must be between 1 and 2000.")
    _, prisma = build_year_stratified_corpus(
        args.topic,
        max_per_year=args.max_per_year,
        output_dir=args.output_dir,
        corpus_name=args.corpus_name,
        prisma_name=args.prisma_name,
    )
    print(json.dumps(prisma, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
