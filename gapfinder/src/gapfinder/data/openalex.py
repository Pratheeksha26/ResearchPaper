from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests

from gapfinder.config import EMAIL, HTTP_CACHE_PATH, OPENALEX_BASE_URL, OPENALEX_API_KEY


DEFAULT_OPENALEX_BASE = OPENALEX_BASE_URL
TOPIC_PHRASES = (
    "graph neural network",
    "GNN",
    "graph convolutional network",
    "graph attention network",
    "message passing neural network",
    "graph representation learning",
)
TOPIC_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgraph neural networks?\b",
        r"\bGNNs?\b",
        r"\bgraph convolutional networks?\b",
        r"\bgraph attention networks?\b",
        r"\bmessage[- ]passing neural networks?\b",
        r"\bgraph representation learning\b",
    )
)


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "gapfinder/0.1 (mailto:research@example.com)"}
    if EMAIL:
        headers["User-Agent"] = f"gapfinder/0.1 ({EMAIL})"
    if False:
        headers["Authorization"] = f"Bearer {OPENALEX_API_KEY}"
    return headers


def _load_cache() -> dict[str, Any]:
    if not HTTP_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(HTTP_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    HTTP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTTP_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _fetch_json(url: str, *, use_cache: bool = True) -> dict[str, Any]:
    cached_responses = _load_cache()
    if use_cache and url in cached_responses:
        return cached_responses[url]

    attempt = 0
    while True:
        try:
            headers = {"User-Agent": f"gapfinder/0.1 ({EMAIL})"}
            params = {"api_key": OPENALEX_API_KEY} if OPENALEX_API_KEY else None
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else (2 ** attempt + random.random())
                time.sleep(delay)
                attempt += 1
                if attempt > 6:
                    raise requests.HTTPError(f"OpenAlex rate limited after {attempt} retries")
                continue
            if response.status_code >= 500:
                delay = 2 ** attempt + random.random()
                time.sleep(delay)
                attempt += 1
                if attempt > 5:
                    raise requests.HTTPError(f"OpenAlex server error: {response.status_code}")
                continue
            response.raise_for_status()
            payload = response.json()
            if use_cache:
                cached_responses[url] = payload
                _save_cache(cached_responses)
            return payload
        except requests.RequestException as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
            if attempt >= 5:
                raise
            delay = 2 ** attempt + random.random()
            time.sleep(delay)
            attempt += 1


def get_rate_limit() -> dict[str, Any]:
    """Fetch the live authenticated OpenAlex budget; this response is intentionally uncached."""
    return _fetch_json(f"{DEFAULT_OPENALEX_BASE}/rate-limit", use_cache=False)


def _fixture_records(topic: str, *, since: int, max_results: int) -> list[dict[str, Any]]:
    base = (topic or "research gap").strip() or "research gap"
    titles = [
        f"{base.title()} and evidence-based discovery",
        f"Structured retrieval for {base}",
        f"A benchmark for {base} evaluation",
        f"Scalable methods for {base}",
        f"Open challenges in {base}",
    ]
    records: list[dict[str, Any]] = []
    for idx, title in enumerate(titles[: max_results]):
        records.append(
            {
                "id": f"fixture:{idx}",
                "title": title,
                "doi": f"https://doi.org/10.1000/{idx + 1}",
                "year": max(since, 2015 + idx),
                "venue": "Fixture benchmark",
                "concepts": [base, "retrieval", "evidence"],
                "referenced_works": [],
                "cited_by_count": idx + 1,
                "abstract": f"This paper studies {base} using a reproducible evidence pipeline with deterministic retrieval and evaluation for {base}.",
                "url": f"https://example.invalid/{idx}",
                "synthetic": True,
            }
        )
    return records


def build_works_url(
    topic: str,
    *,
    since: int,
    per_page: int,
    page: int,
    publication_year: int | None = None,
) -> str:
    """Build an OpenAlex works URL using documented date and English filters."""
    year_filter = (
        f"publication_year:{publication_year}"
        if publication_year is not None
        else f"from_publication_date:{since:04d}-01-01"
    )
    params = {
        "search": topic,
        "filter": f"{year_filter},language:en",
        "per_page": per_page,
        "page": page,
        "sort": "relevance_score:desc",
    }
    return f"{DEFAULT_OPENALEX_BASE}/works?{urlencode(params)}"


def build_phrase_works_url(*, publication_year: int, per_page: int, page: int) -> str:
    """Build a year-filtered exact-phrase search; title/abstract are verified locally."""
    phrase_query = "(" + " OR ".join(f'"{phrase}"' for phrase in TOPIC_PHRASES) + ")"
    params = {
        "search": phrase_query,
        "filter": f"publication_year:{publication_year},language:en",
        "per_page": per_page,
        "page": page,
        "sort": "relevance_score:desc",
    }
    return f"{DEFAULT_OPENALEX_BASE}/works?{urlencode(params)}"


def _paper_record(result: dict[str, Any]) -> dict[str, Any]:
    location = result.get("primary_location") or {}
    source = location.get("source") or {}
    ids = result.get("ids") or {}
    return {
        "id": result.get("id"),
        "title": result.get("title"),
        "doi": result.get("doi"),
        "year": result.get("publication_year") or result.get("year"),
        "venue": source.get("display_name"),
        "concepts": [
            concept.get("display_name")
            for concept in result.get("concepts", [])
            if concept.get("display_name")
        ],
        "referenced_works": result.get("referenced_works", []),
        "cited_by_count": result.get("cited_by_count", 0),
        "abstract": reconstruct_abstract(result),
        "url": ids.get("openalex"),
    }


def search_openalex_phrase_with_metadata(
    *,
    publication_year: int,
    max_results: int = 200,
    per_page: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return records whose title or abstract contains a requested topic phrase.

    OpenAlex's current phrase search spans title, abstract, and full text. Results
    are therefore checked against title and reconstructed abstract before they
    enter the candidate pool.
    """
    if not 1 <= per_page <= 100:
        raise ValueError("OpenAlex per_page must be between 1 and 100.")
    if not 1 <= max_results <= 2000:
        raise ValueError("max_results must be between 1 and 2000.")

    records: list[dict[str, Any]] = []
    total_matches: int | None = None
    scanned = 0
    page = 1
    while len(records) < max_results and page <= 50:
        payload = _fetch_json(
            build_phrase_works_url(
                publication_year=publication_year,
                per_page=per_page,
                page=page,
            )
        )
        if total_matches is None:
            total_matches = (payload.get("meta") or {}).get("count")
        results = payload.get("results", [])
        if not results:
            break
        for result in results:
            scanned += 1
            record = _paper_record(result)
            text = f"{record.get('title') or ''}\n{record.get('abstract') or ''}"
            if any(pattern.search(text) for pattern in TOPIC_PATTERNS):
                records.append(record)
                if len(records) >= max_results:
                    break
        page += 1

    return records, {
        "total_matches": total_matches,
        "sort": "relevance_score:desc",
        "filter": f"publication_year:{publication_year},language:en",
        "search_parameter": "search with quoted phrase OR terms",
        "search_phrases": list(TOPIC_PHRASES),
        "title_abstract_phrase_required": True,
        "api_results_scanned": scanned,
        "pages_fetched": page - 1,
        "language_filter": "language:en",
    }


def search_openalex(topic: str, *, since: int = 2015, per_page: int = 100, max_results: int = 500, fixture: bool = False) -> list[dict[str, Any]]:
    records, _metadata = search_openalex_with_metadata(
        topic, since=since, per_page=per_page, max_results=max_results, fixture=fixture
    )
    return records


def search_openalex_with_metadata(
    topic: str,
    *,
    since: int = 2015,
    per_page: int = 100,
    max_results: int = 500,
    fixture: bool = False,
    publication_year: int | None = None,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search OpenAlex by topic and return paper-like records with reconstructed abstracts.

    The default path fails loudly when the live OpenAlex API is unavailable. A fixture corpus is
    only allowed for tests or explicit `--fixture` invocations.
    """
    if fixture:
        return _fixture_records(topic, since=since, max_results=max_results), {
            "total_matches": None,
            "sort": "fixture order",
            "language_filter": None,
            "filter": None,
        }

    if not 1 <= per_page <= 100:
        raise ValueError("OpenAlex per_page must be between 1 and 100.")
    if not 1 <= max_results <= 2000:
        raise ValueError("max_results must be between 1 and 2000.")
    items: list[dict[str, Any]] = []
    page = 1
    total_matches: int | None = None

    while len(items) < max_results:
        url = build_works_url(
            topic,
            since=since,
            per_page=per_page,
            page=page,
            publication_year=publication_year,
        )
        payload = _fetch_json(url)
        if total_matches is None:
            meta = payload.get("meta") or {}
            total_matches = meta.get("count")
        results = payload.get("results", [])
        if not results:
            break
        for result in results:
            location = result.get("primary_location") or {}
            source = location.get("source") or {}
            ids = result.get("ids") or {}
            record = {
                "id": result.get("id"),
                "title": result.get("title"),
                "doi": result.get("doi"),
                "year": result.get("publication_year") or result.get("year"),
                "venue": source.get("display_name"),
                "concepts": [concept.get("display_name") for concept in result.get("concepts", []) if concept.get("display_name")],
                "referenced_works": result.get("referenced_works", []),
                "cited_by_count": result.get("cited_by_count", 0),
                "abstract": reconstruct_abstract(result),
                "url": ids.get("openalex"),
            }
            items.append(record)
            if len(items) >= max_results:
                break
        page += 1
        if page > 50:
            break

    if not items and not allow_empty:
        raise RuntimeError(f"OpenAlex returned no results for topic {topic!r}; no synthetic fallback is allowed in the default path.")

    return items, {
        "total_matches": total_matches,
        "sort": "relevance_score:desc",
        "language_filter": "language:en",
        "filter": (
            f"publication_year:{publication_year},language:en"
            if publication_year is not None
            else f"from_publication_date:{since:04d}-01-01,language:en"
        ),
    }


def reconstruct_abstract(result: dict[str, Any]) -> str:
    """Reconstruct a readable abstract from OpenAlex's inverted index.

    The inverted index stores multiple positions per token; for readability we keep a single
    representative occurrence per token at its earliest position instead of repeating the token
    once for every occurrence. This keeps the reconstruction stable and avoids duplicated words.
    """
    abstract = result.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract

    abstract_inverted_index = result.get("abstract_inverted_index")
    if not abstract_inverted_index:
        return ""

    indexed_tokens: dict[int, str] = {}
    for token, positions in abstract_inverted_index.items():
        if not positions:
            continue
        for raw_position in positions:
            position = int(raw_position)
            if position < 0:
                raise ValueError("Abstract inverted-index positions must be nonnegative.")
            existing = indexed_tokens.get(position)
            if existing is not None and existing != token:
                raise ValueError(
                    f"Abstract inverted index assigns position {position} to multiple tokens."
                )
            indexed_tokens[position] = token

    if not indexed_tokens:
        return ""
    return " ".join(indexed_tokens[position] for position in sorted(indexed_tokens))


def export_records(records: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


__all__ = [
    "search_openalex",
    "search_openalex_with_metadata",
    "build_works_url",
    "get_rate_limit",
    "reconstruct_abstract",
    "export_records",
]
