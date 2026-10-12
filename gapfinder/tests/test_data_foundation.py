from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from gapfinder.data.dedupe import dedupe_records, normalize_title
from gapfinder.data import openalex
from gapfinder.data.build_year_stratified import _cap_records_per_year
from gapfinder.data.build import _safe_request_url
from gapfinder.data import build as build_module
from gapfinder.data.openalex import (
    TOPIC_PHRASES,
    build_phrase_works_url,
    build_works_url,
    reconstruct_abstract,
)
from gapfinder.data.prisma import make_prisma_counts
from gapfinder import config


def test_env_loader_strips_whitespace_and_quotes_without_overriding_environment(tmp_path: Path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text('OPENALEX_API_KEY = "  file-key  "\nGAPFINDER_MAILTO= mail@example.org \n', encoding="utf-8")
    monkeypatch.delenv("GAPFINDER_MAILTO", raising=False)
    monkeypatch.setenv("OPENALEX_API_KEY", "  inherited-key  ")

    config._load_dotenv(dotenv_path)

    assert config._clean_env_value("  ' quoted-key '  ") == "quoted-key"
    assert config._clean_env_value(' "quoted-key" ') == "quoted-key"
    assert config._clean_env_value(" plain-key \n") == "plain-key"
    assert os.environ["OPENALEX_API_KEY"] == "  inherited-key  "
    assert os.environ["GAPFINDER_MAILTO"] == "mail@example.org"


def test_gemini_secret_refreshes_from_dotenv_without_echoing(
    tmp_path: Path, monkeypatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("ROTATABLE_TEST_KEY=old-value\n", encoding="utf-8")
    monkeypatch.setattr(config, "DOTENV_PATH", dotenv)
    monkeypatch.setattr(config, "_PREEXISTING_ENV", set())
    monkeypatch.setitem(config._DOTENV_INITIAL_VALUES, "ROTATABLE_TEST_KEY", "old-value")
    monkeypatch.setenv("ROTATABLE_TEST_KEY", "old-value")
    assert config.get_secret("ROTATABLE_TEST_KEY") == "old-value"
    assert config.get_secret_source("ROTATABLE_TEST_KEY") == "dotenv"

    dotenv.write_text("ROTATABLE_TEST_KEY='  new-value  '\n", encoding="utf-8")
    assert config.get_secret("ROTATABLE_TEST_KEY") == "new-value"


def test_inherited_secret_still_takes_precedence_over_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("ROTATABLE_TEST_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setattr(config, "DOTENV_PATH", dotenv)
    monkeypatch.setattr(config, "_PREEXISTING_ENV", {"ROTATABLE_TEST_KEY"})
    monkeypatch.setenv("ROTATABLE_TEST_KEY", "inherited-value")
    assert config.get_secret("ROTATABLE_TEST_KEY") == "inherited-value"
    assert config.get_secret_source("ROTATABLE_TEST_KEY") == "environment"


def test_openalex_sends_api_key_as_query_parameter(monkeypatch) -> None:
    captured = {}

    class Response:
        status_code = 200
        headers = {}

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"ok": True}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(openalex, "OPENALEX_API_KEY", "test-key-123")
    monkeypatch.setattr(openalex.requests, "get", fake_get)

    assert openalex._fetch_json("https://api.openalex.org/rate-limit", use_cache=False) == {"ok": True}
    assert captured["params"] == {"api_key": "test-key-123"}
    assert "Authorization" not in captured["headers"]


def test_error_request_url_redacts_api_key() -> None:
    request = openalex.requests.Request(
        "GET",
        "https://api.openalex.org/rate-limit",
        params={"api_key": "test-key-123"},
    ).prepare()
    response = SimpleNamespace(request=request)

    safe_url = _safe_request_url(response)

    assert safe_url == "https://api.openalex.org/rate-limit?api_key=***"
    assert "test-key-123" not in safe_url


def test_reconstruct_abstract_from_inverted_index() -> None:
    result = {
        "abstract_inverted_index": {
            "research": [0, 4],
            "gap": [1, 5],
            "detection": [2],
            "with": [3],
        }
    }
    assert reconstruct_abstract(result) == "research gap detection with research gap"


def test_reconstruct_abstract_preserves_every_repeated_token_occurrence() -> None:
    result = {
        "abstract_inverted_index": {
            "A": [0],
            "model": [1],
            "uses": [2],
            "graph": [3, 7],
            "neural": [4],
            "networks": [5],
            "for": [6],
            "classification.": [8],
        }
    }
    assert (
        reconstruct_abstract(result)
        == "A model uses graph neural networks for graph classification."
    )


def test_openalex_works_url_uses_documented_from_publication_date_filter() -> None:
    from urllib.parse import parse_qs, urlsplit

    url = build_works_url("graph neural networks", since=2015, per_page=100, page=1)
    query = parse_qs(urlsplit(url).query)

    assert query["filter"] == ["from_publication_date:2015-01-01,language:en"]
    assert query["sort"] == ["relevance_score:desc"]
    assert query["per_page"] == ["100"]
    assert "9999" not in url


def test_openalex_works_url_uses_exact_publication_year_filter() -> None:
    from urllib.parse import parse_qs, urlsplit

    url = build_works_url(
        "graph neural networks",
        since=2015,
        publication_year=2024,
        per_page=100,
        page=1,
    )
    query = parse_qs(urlsplit(url).query)
    assert query["filter"] == ["publication_year:2024,language:en"]


def test_phrase_query_uses_quoted_terms_and_fixed_year_filters() -> None:
    from urllib.parse import parse_qs, urlsplit

    url = build_phrase_works_url(publication_year=2017, per_page=100, page=2)
    query = parse_qs(urlsplit(url).query)

    assert query["filter"] == ["publication_year:2017,language:en"]
    assert query["sort"] == ["relevance_score:desc"]
    assert query["per_page"] == ["100"]
    assert query["page"] == ["2"]
    assert all(f'"{phrase}"' in query["search"][0] for phrase in TOPIC_PHRASES)
    assert '"message passing"' not in query["search"][0]
    assert '"message passing neural network"' in query["search"][0]
    assert "graph neural networks" not in query["search"][0]


def test_phrase_search_excludes_full_text_only_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        openalex,
        "_fetch_json",
        lambda _url: {
            "meta": {"count": 3},
            "results": [
                {
                    "id": "W1",
                    "title": "Graph Neural Network for Citation Prediction",
                    "publication_year": 2017,
                    "abstract": "We predict links in graphs.",
                },
                {
                    "id": "W2",
                    "title": "Neural Networks for Images",
                    "publication_year": 2017,
                    "abstract": "An unrelated abstract.",
                },
            ],
        },
    )

    records, metadata = openalex.search_openalex_phrase_with_metadata(
        publication_year=2017, max_results=1
    )

    assert [record["id"] for record in records] == ["W1"]
    assert metadata["total_matches"] == 3
    assert metadata["title_abstract_phrase_required"] is True
    assert metadata["api_results_scanned"] == 1


def test_strict_phrase_filter_rejects_unqualified_message_passing(monkeypatch) -> None:
    monkeypatch.setattr(
        openalex,
        "_fetch_json",
        lambda _url: {
            "meta": {"count": 3},
            "results": [
                {
                    "id": "W1",
                    "title": "Transaction flows and message-passing protocols",
                    "publication_year": 2017,
                    "abstract": "Formal methods for executable distributed protocols.",
                },
                {
                    "id": "W2",
                    "title": "Message Passing Neural Networks for Molecules",
                    "publication_year": 2017,
                    "abstract": "A neural network predicts molecular properties.",
                },
                {
                    "id": "W3",
                    "title": "Graph Neural Network for Citation Prediction",
                    "publication_year": 2017,
                    "abstract": "We predict links in graphs.",
                },
            ],
        },
    )

    records, _metadata = openalex.search_openalex_phrase_with_metadata(
        publication_year=2017, max_results=2
    )

    assert [record["id"] for record in records] == ["W2", "W3"]


def test_yearly_included_cap_preserves_order_and_counts_removed() -> None:
    records = [
        {"id": "A", "year": 2020},
        {"id": "B", "year": 2020},
        {"id": "C", "year": 2021},
    ]

    included, removed = _cap_records_per_year(records, limit=1)

    assert [record["id"] for record in included] == ["A", "C"]
    assert removed[2020] == 1


def test_openalex_handles_null_nested_metadata(monkeypatch) -> None:
    monkeypatch.setattr(openalex, "_fetch_json", lambda _url: {"results": [
        {
            "id": "https://openalex.org/W1",
            "title": "Paper with sparse metadata",
            "primary_location": None,
            "ids": None,
            "concepts": [],
            "referenced_works": [],
            "abstract_inverted_index": {"A": [0], "useful": [1], "abstract": [2], "has": [3], "words": [4]},
        }
    ]})

    records = openalex.search_openalex("test", max_results=1)

    assert records[0]["venue"] is None
    assert records[0]["url"] is None


def test_dedupe_records_by_doi_and_title() -> None:
    records = [
        {"title": "Graph Neural Networks for Drug Discovery", "doi": "https://doi.org/10.1234/example"},
        {"title": "Graph Neural Networks for Drug Discovery ", "doi": "https://doi.org/10.1234/example"},
        {"title": "Graph Neural Networks for Drug Discovery!", "doi": "https://doi.org/10.9999/other"},
        {"title": "Different Paper", "doi": "https://doi.org/10.5555/other"},
    ]
    deduped = dedupe_records(records)
    assert len(deduped) == 2
    assert normalize_title("Graph Neural Networks for Drug Discovery!") == "graph neural networks for drug discovery"


def test_embedding_dedupe_reports_removed_records() -> None:
    records = [
        {"title": "Paper A", "embedding": [1.0, 0.0]},
        {"title": "Paper B", "embedding": [0.99, 0.01]},
        {"title": "Paper C", "embedding": [0.0, 1.0]},
    ]
    deduped, counts = dedupe_records(records, return_stats=True)
    assert len(deduped) == 2
    assert counts == {"doi_exact": 0, "title_normalized": 0, "embedding_cosine": 1}


def test_prisma_counts_are_deterministic() -> None:
    counts = make_prisma_counts([
        {"title": "A", "year": 2024, "abstract": "This is a valid abstract with enough content."},
        {"title": "B", "year": 2024, "abstract": "This is also a valid abstract with enough content."},
    ])
    assert counts == {
        "identified": 2,
        "deduped": 2,
        "abstract_missing_removed": 0,
        "after_abstract_missing_removal": 2,
        "language_date_filtered": 0,
        "after_language_date_filtering": 2,
        "included": 2,
    }


def test_built_corpus_matches_prisma_included_count_and_writes_exclusions(tmp_path: Path, monkeypatch) -> None:
    records = [
        {"id": "W1", "title": "Valid Paper", "year": 2024, "abstract": "This paper has a sufficiently long English abstract."},
        {"id": "W2", "title": "Missing Abstract", "year": 2024, "abstract": ""},
    ]
    monkeypatch.setattr(
        build_module,
        "search_openalex_with_metadata",
        lambda *args, **kwargs: (records, {
            "total_matches": 2,
            "sort": "relevance_score:desc",
            "language_filter": "language:en",
        }),
    )
    monkeypatch.setattr(build_module, "get_rate_limit", lambda: {})

    corpus, prisma = build_module.build_corpus("test", max_results=2, output_dir=tmp_path)
    corpus_lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    excluded = [
        json.loads(line)
        for line in (tmp_path / "excluded.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(corpus) == prisma["included"] == len(corpus_lines) == 1
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reason"] == "abstract_missing"
    assert prisma["embedding_dedupe_status"] == "skipped"


def test_year_stratified_build_records_year_counts_and_matches_prisma(
    tmp_path: Path, monkeypatch
) -> None:
    from gapfinder.data import build_year_stratified

    calls: list[int] = []

    def fake_search(*, publication_year: int, **_kwargs):
        calls.append(publication_year)
        rows = (
            [
                {
                    "id": f"W{publication_year}",
                    "title": f"Paper {publication_year}",
                    "year": publication_year,
                    "abstract": "A sufficiently long English abstract for a paper.",
                    "synthetic": False,
                }
            ]
            if publication_year == 2024
            else []
        )
        return rows, {
            "total_matches": 8 if publication_year == 2024 else 0,
            "sort": "relevance_score:desc",
            "filter": f"publication_year:{publication_year},language:en",
            "search_parameter": "search with quoted phrase OR terms",
            "search_phrases": ["graph neural network"],
            "title_abstract_phrase_required": True,
            "api_results_scanned": 1 if publication_year == 2024 else 0,
            "pages_fetched": 1 if publication_year == 2024 else 0,
            "language_filter": "language:en",
        }

    monkeypatch.setattr(
        build_year_stratified, "search_openalex_phrase_with_metadata", fake_search
    )
    monkeypatch.setattr(build_year_stratified, "get_rate_limit", lambda: {})
    corpus, prisma = build_year_stratified.build_year_stratified_corpus(
        "test", output_dir=tmp_path
    )

    assert calls == list(range(2015, 2027))
    assert len(corpus) == prisma["included"] == 1
    assert prisma["per_year"]["2024"]["identified"] == 1
    assert prisma["per_year"]["2024"]["included"] == 1
    assert prisma["per_year"]["2024"]["total_matches"] == 8
    assert prisma["openalex_total_matches_sum_by_year"] == 8
