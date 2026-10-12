from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from gapfinder.data.dedupe import dedupe_records
from gapfinder.config import OPENALEX_API_KEY
from gapfinder.data.openalex import export_records, get_rate_limit, search_openalex_with_metadata
from gapfinder.data.prisma import export_flow_diagram, export_prisma, make_prisma_counts, screen_records


def _budget_summary(payload: dict) -> dict:
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


def _safe_error_body(response: requests.Response) -> str:
    body = response.text
    if OPENALEX_API_KEY:
        body = body.replace(OPENALEX_API_KEY, "[REDACTED]")
    body = re.sub(r'("api_key"\s*:\s*)"[^"]*"', r'\1"[REDACTED]"', body)
    return body


def _safe_request_url(response: requests.Response) -> str:
    request = response.request
    if request is None or request.url is None:
        return "[request URL unavailable]"
    parsed = urlsplit(request.url)
    query = [
        (name, "***" if name == "api_key" else value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    encoded_query = urlencode(query).replace("api_key=%2A%2A%2A", "api_key=***")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, encoded_query, parsed.fragment))


def build_corpus(
    topic: str,
    *,
    since: int = 2015,
    max_results: int = 500,
    output_dir: str | Path = ".",
    fixture: bool = False,
    corpus_name: str = "corpus.jsonl",
    prisma_name: str = "prisma.json",
) -> tuple[list[dict], dict]:
    if not fixture:
        print(f"OpenAlex rate limit at build start: {json.dumps(_budget_summary(get_rate_limit()), sort_keys=True)}")
    try:
        records, search_metadata = search_openalex_with_metadata(
            topic, since=since, max_results=max_results, per_page=100, fixture=fixture
        )
        dedupe_result = dedupe_records(records, return_stats=True)
        if not isinstance(dedupe_result, tuple):
            raise TypeError("Deduplication did not return its requested stage counts.")
        unique, dedupe_counts = dedupe_result
        prisma = make_prisma_counts(unique)
        included, excluded, screening_counts = screen_records(unique)
        prisma["identified"] = len(records)
        prisma["after_doi_dedupe"] = len(records) - dedupe_counts["doi_exact"]
        prisma["after_title_dedupe"] = prisma["after_doi_dedupe"] - dedupe_counts["title_normalized"]
        prisma["after_embedding_dedupe"] = len(unique)
        prisma["embedding_dedupe_status"] = (
            "applied" if any(isinstance(record.get("embedding"), list) for record in unique) else "skipped"
        )
        prisma["abstract_missing_removed"] = screening_counts["abstract_missing"]
        prisma["after_abstract_missing_removal"] = len(unique) - screening_counts["abstract_missing"]
        prisma["language_date_filtered"] = screening_counts["language_date_filtered"]
        prisma["after_language_date_filtering"] = len(included)
        prisma["included"] = len(included)
        prisma["openalex_total_matches"] = search_metadata["total_matches"]
        prisma["sort_order"] = search_metadata["sort"]
        prisma["language_filter_applied"] = search_metadata["language_filter"] == "language:en"
        print(f"Record counts by stage: {json.dumps({key: prisma[key] for key in ('identified', 'after_doi_dedupe', 'after_title_dedupe', 'after_embedding_dedupe', 'after_abstract_missing_removal', 'after_language_date_filtering', 'included')}, sort_keys=True)}")

        if any(record.get("synthetic") is True for record in unique):
            raise ValueError("Synthetic records cannot be written as a paper-run corpus or PRISMA report.")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if len(included) != prisma["included"]:
            raise AssertionError("Included corpus length does not match the PRISMA included count.")
        export_records(included, output / corpus_name)
        export_records(excluded, output / "excluded.jsonl")
        if not fixture:
            export_prisma(prisma, output / prisma_name)
        export_flow_diagram(prisma, output / "prisma_flow.md")

        return included, prisma
    finally:
        if not fixture:
            print(f"OpenAlex rate limit at build end: {json.dumps(_budget_summary(get_rate_limit()), sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic evidence-grounded corpus for a topic.")
    parser.add_argument("--topic", required=True, help="Research topic to query.")
    parser.add_argument("--since", type=int, default=2015, help="Minimum publication year.")
    parser.add_argument("--max", type=int, default=500, help="Maximum number of records to retrieve (1-2000).")
    parser.add_argument("--output-dir", default=".", help="Directory to write corpus and PRISMA outputs.")
    parser.add_argument("--corpus-name", default="corpus.jsonl", help="Included-corpus filename.")
    parser.add_argument("--prisma-name", default="prisma.json", help="PRISMA-report filename.")
    parser.add_argument("--fixture", action="store_true", help="Use the in-repo fixture corpus only for tests. Synthetic runs are refused for report output.")
    args = parser.parse_args()

    if not 1 <= args.max <= 2000:
        parser.error("--max must be between 1 and 2000.")
    try:
        _, prisma = build_corpus(
            args.topic,
            since=args.since,
            max_results=args.max,
            output_dir=args.output_dir,
            fixture=args.fixture,
            corpus_name=args.corpus_name,
            prisma_name=args.prisma_name,
        )
    except requests.RequestException as exc:
        status_code = exc.response.status_code if exc.response is not None else "unavailable"
        print(f"OpenAlex API error: HTTP {status_code}", file=sys.stderr)
        if exc.response is not None:
            print(f"OpenAlex request URL: {_safe_request_url(exc.response)}", file=sys.stderr)
            print(_safe_error_body(exc.response), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(prisma, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
