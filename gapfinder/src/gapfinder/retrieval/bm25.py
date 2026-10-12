from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize text using lowercase alphanumeric terms."""
    return TOKEN_RE.findall(text.lower())


def document_text(record: dict[str, Any]) -> str:
    """Build a retrieval document from a record's title and abstract."""
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    return f"{title} {title} {abstract}".strip()


@dataclass(frozen=True)
class BM25Index:
    """A small, dependency-free BM25 index suitable for research corpora."""

    documents: tuple[tuple[str, ...], ...]
    average_length: float
    document_frequency: dict[str, int]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_documents(
        cls, documents: Iterable[str], *, k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        """Create an index from document strings."""
        tokenized = tuple(tuple(tokenize(text)) for text in documents)
        frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            frequencies.update(set(tokens))
        average_length = (
            sum(map(len, tokenized)) / len(tokenized) if tokenized else 0.0
        )
        return cls(tokenized, average_length, dict(frequencies), k1, b)

    def scores(self, query: str) -> list[float]:
        """Score each indexed document for a query using BM25."""
        query_terms = tokenize(query)
        count = len(self.documents)
        if not count:
            return []

        scores: list[float] = []
        for tokens in self.documents:
            term_counts = Counter(tokens)
            score = 0.0
            length_norm = (
                self.k1 * (1.0 - self.b + self.b * len(tokens) / self.average_length)
                if self.average_length
                else self.k1
            )
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1.0 + (count - df + 0.5) / (df + 0.5)
                )
                score += inverse_frequency * (
                    frequency * (self.k1 + 1.0) / (frequency + length_norm)
                )
            scores.append(score)
        return scores

    def rank(self, query: str, *, limit: int | None = None) -> list[tuple[int, float]]:
        """Return ``(document_index, score)`` pairs, highest score first."""
        results = sorted(
            enumerate(self.scores(query)), key=lambda item: (-item[1], item[0])
        )
        return results if limit is None else results[:limit]


def rank_documents(
    query: str,
    documents: Sequence[str],
    *,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Index and rank document strings for one query."""
    return BM25Index.from_documents(documents).rank(query, limit=limit)
