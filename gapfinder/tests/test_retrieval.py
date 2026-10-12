from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gapfinder.eval.survey_recovery import (
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    run_evaluation,
    select_surveys,
)
from gapfinder.eval import survey_recovery
from gapfinder.retrieval.bm25 import BM25Index, document_text
from gapfinder.retrieval.dense import DenseRetriever, configured_model_name
from gapfinder.retrieval.fusion import reciprocal_rank_fusion


def test_bm25_ranks_matching_document_and_uses_title_twice() -> None:
    records = [
        {"title": "Graph neural network methods", "abstract": "Message passing."},
        {"title": "Unrelated topic", "abstract": "No graph models here."},
    ]
    index = BM25Index.from_documents([document_text(row) for row in records])

    assert index.rank("graph neural network")[0][0] == 0
    assert document_text(records[0]).startswith("Graph neural network methods Graph neural network methods")


def test_rrf_uses_one_based_ranks_and_deduplicates_each_input() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": ["a", "a", "b"], "dense": ["b", "a"]}, k=60
    )

    assert fused[0] == ("a", pytest.approx(1 / 61 + 1 / 62))
    assert fused[1] == ("b", pytest.approx(1 / 62 + 1 / 61))


def test_weighted_rrf_preserves_unweighted_default() -> None:
    rankings = {"bm25": ["a", "b"], "dense": ["b", "a"]}
    assert reciprocal_rank_fusion(rankings, k=60) == reciprocal_rank_fusion(
        rankings, k=60, weights={"bm25": 1.0, "dense": 1.0}
    )
    weighted = reciprocal_rank_fusion(
        rankings, k=60, weights={"bm25": 1.0, "dense": 0.0}
    )
    assert weighted[0][0] == "a"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, weights={"dense": -1.0})


class ToyEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences, **_kwargs):
        self.calls += 1
        vectors = {
            "graph": [1.0, 0.0],
            "neural": [0.0, 1.0],
            "graph query": [1.0, 0.0],
        }
        return np.asarray([vectors[item] for item in sentences], dtype=np.float32)


def test_dense_retriever_persists_embeddings_and_ranks_with_cosine(tmp_path: Path) -> None:
    encoder = ToyEncoder()
    first = DenseRetriever("toy-model", cache_dir=tmp_path, encoder=encoder)
    first.fit(["graph", "neural"])
    assert first.rank("graph query")[0][0] == 0
    assert len(list(tmp_path.glob("*.npz"))) == 1

    cached = DenseRetriever(
        "toy-model", cache_dir=tmp_path, encoder=ToyEncoder()
    ).fit(["graph", "neural"])
    assert cached.rank("graph query")[0][0] == 0
    assert encoder.calls == 2  # corpus and query only; cached fit did not encode.
    assert not first.used_faiss


def test_faiss_backend_is_explicit_and_functional_when_installed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    retriever = DenseRetriever(
        "toy-model",
        cache_dir=tmp_path,
        encoder=ToyEncoder(),
        backend="faiss",
    ).fit(["graph", "neural"])
    assert retriever.used_faiss
    assert retriever.rank("graph query")[0][0] == 0


def test_bootstrap_ci_and_exact_paired_wilcoxon_are_deterministic() -> None:
    first = survey_recovery._bootstrap_mean_ci([0.0, 0.5, 1.0], seed=42)
    assert first == survey_recovery._bootstrap_mean_ci([0.0, 0.5, 1.0], seed=42)
    assert first[0] == pytest.approx(0.5)
    assert survey_recovery._paired_wilcoxon_pvalue([1, 2], [0, 1]) == pytest.approx(0.5)
    assert survey_recovery._paired_wilcoxon_pvalue([1, 1], [1, 1]) == 1.0


def test_retrieval_metrics_are_binary_and_skip_empty_gold_sets() -> None:
    ranking = ["W1", "W2", "W3"]
    relevant = {"W2", "W3"}

    assert recall_at_k(ranking, relevant, 2) == 0.5
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert ndcg_at_k(ranking, relevant, 2) == pytest.approx(expected)
    assert recall_at_k(ranking, set(), 5) is None
    assert ndcg_at_k(ranking, set(), 5) is None
    assert precision_at_k(ranking, relevant, 2) == 0.5
    assert precision_at_k(ranking, set(), 2) is None
    assert mean_average_precision(ranking, relevant) == pytest.approx((0.5 + 2 / 3) / 2)
    assert mean_average_precision(ranking, set()) is None


def test_holm_adjustment_preserves_order_and_controls_familywise_error() -> None:
    assert survey_recovery._holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    with pytest.raises(ValueError):
        survey_recovery._holm_adjust([-0.1])


def test_survey_selection_uses_only_real_openalex_survey_rows() -> None:
    rows = [
        {
            "id": "https://openalex.org/W100",
            "title": "A survey of graph learning",
            "referenced_works": ["https://openalex.org/W101"],
            "synthetic": True,
        },
        {
            "id": "https://openalex.org/W102",
            "title": "A survey of graph learning",
            "referenced_works": ["https://openalex.org/W103"],
        },
        {
            "id": "fixture:1",
            "title": "A survey of graph learning",
            "referenced_works": ["fixture:2"],
        },
    ]

    assert [row["id"] for row in select_surveys(rows)] == [
        "https://openalex.org/W102"
    ]


def test_fixture_is_explicitly_synthetic_and_never_selected() -> None:
    fixture = Path(__file__).parent / "fixtures" / "retrieval_toy.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]

    assert all(row["synthetic"] for row in rows)
    assert select_surveys(rows) == []


def test_embedding_model_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAPFINDER_EMBEDDING_MODEL", "BAAI/test-model")
    assert configured_model_name() == "BAAI/test-model"


def test_evaluator_ground_truth_uses_only_resolved_referenced_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "https://openalex.org/W100",
            "title": "A survey of graph neural retrieval",
            "year": 2023,
            "abstract": "A survey discusses graph retrieval.",
            "referenced_works": ["https://openalex.org/W101"],
        },
        {
            "id": "https://openalex.org/W101",
            "title": "Graph retrieval methods",
            "year": 2020,
            "abstract": "A real paper on graph retrieval.",
            "referenced_works": [],
        },
        {
            "id": "https://openalex.org/W102",
            "title": "Unrelated methods",
            "year": 2020,
            "abstract": "This paper is unrelated.",
            "referenced_works": [],
        },
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    lookup_payloads = {
        "W100": {
            "id": "https://openalex.org/W100",
            "title": "A survey of graph neural retrieval",
            "publication_year": 2023,
            "abstract": "A survey of graph neural retrieval.",
            "referenced_works": ["https://openalex.org/W101"],
            "ids": {},
        },
        "W101": {
            "id": "https://openalex.org/W101",
            "title": "Graph retrieval methods",
            "publication_year": 2020,
            "referenced_works": [],
            "ids": {},
        },
    }

    class FakeLookup:
        def __init__(self, *_args, **_kwargs):
            pass

        def get(self, work_id):
            return lookup_payloads[work_id]

        def save(self):
            pass

    class FakeDense:
        used_faiss = False

        def __init__(self, *_args, **_kwargs):
            self.documents = []

        def fit(self, documents):
            self.documents = list(documents)
            return self

        def rank(self, query, *, limit=None):
            return [(index, 1.0) for index in range(len(self.documents))]

    monkeypatch.setattr(survey_recovery, "OpenAlexSingletons", FakeLookup)
    monkeypatch.setattr(survey_recovery, "DenseRetriever", FakeDense)
    output = tmp_path / "run"
    result = run_evaluation(
        corpus_path=corpus,
        output_dir=output,
        max_surveys=1,
        min_relevant_documents=1,
        cutoffs=(1,),
        model_name="test-model",
    )

    gold = json.loads((output / "ground_truth.jsonl").read_text(encoding="utf-8"))
    metrics = list(
        __import__("csv").DictReader(
            (output / "metrics.csv").open(encoding="utf-8-sig")
        )
    )
    assert result["ground_truth_relevant_candidate_count"] == 1
    assert gold["relevant_candidate_ids"] == ["W101"]
    assert "W102" not in gold["relevant_candidate_ids"]
    assert all(
        row["status"] == "ok"
        for row in metrics
        if row["method"] in {"BM25", "dense", "RRF"}
    )
    assert all(
        row["status"] == "LOO tuning requires at least two qualified queries"
        for row in metrics
        if row["method"] == "weighted_RRF_LOO"
    )
    assert {
        row["metric"]
        for row in metrics
        if row["method"] == "BM25"
    } >= {"Precision@1", "MAP"}
    paired = list(
        __import__("csv").DictReader(
            (output / "paired_tests.csv").open(encoding="utf-8-sig")
        )
    )
    assert len(paired) == 6
    assert all(row["holm_p_all_10"] for row in paired)
    assert all(
        row["holm_p_dense_vs_bm25_5"]
        for row in paired
        if row["right_method"] == "BM25"
    )
    unresolved = list(
        __import__("csv").DictReader(
            (output / "unresolved_by_survey.csv").open(encoding="utf-8-sig")
        )
    )
    assert unresolved[0]["total_refs"] == "1"
    assert unresolved[0]["unresolved"] == "0"
    assert all(
        "W100" not in [
            hit["work_id"] for hit in trial["ranking"]
        ]
        for trial in survey_recovery._read_jsonl(output / "trials_bm25.jsonl")
    )
