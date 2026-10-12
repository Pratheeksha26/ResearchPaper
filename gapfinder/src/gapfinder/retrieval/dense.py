from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import numpy as np

from gapfinder.config import CACHE_DIR

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL_ENV = "GAPFINDER_EMBEDDING_MODEL"


class Encoder(Protocol):
    """Minimal interface required from a sentence embedding model."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


def configured_model_name() -> str:
    """Return the fixed scientific embedding model selected through the environment."""
    import os

    return (
        os.environ.get(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:100] or "model"


def _text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_encoder(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires the optional dependencies; install "
            "`pip install -e .[retrieval]`."
        ) from exc
    return cast(Encoder, SentenceTransformer(model_name))


class DenseRetriever:
    """Exact cosine-similarity dense retrieval with a persistent embedding cache.

    NumPy exact search is the default. FAISS ``IndexFlatIP`` can be explicitly
    selected when the optional FAISS package is installed.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        encoder: Encoder | None = None,
        backend: str = "numpy",
    ) -> None:
        if backend not in {"numpy", "faiss"}:
            raise ValueError("Dense backend must be 'numpy' or 'faiss'.")
        self.model_name = model_name or configured_model_name()
        self.cache_dir = Path(cache_dir or CACHE_DIR / "retrieval")
        self._encoder = encoder
        self.backend = backend
        self._index: Any = None
        self._matrix: np.ndarray | None = None
        self.used_faiss = False

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = _load_encoder(self.model_name)
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(
            self.encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding model returned an unexpected vector matrix.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def fit(self, documents: Sequence[str]) -> DenseRetriever:
        """Embed documents once and cache vectors by model and content hash."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        content_hash = _text_hash(documents)
        cache_path = self.cache_dir / (
            f"{_safe_component(self.model_name)}-{content_hash}.npz"
        )
        vectors: np.ndarray | None = None
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as saved:
                    if (
                        str(saved["model"].item()) == self.model_name
                        and str(saved["content_hash"].item()) == content_hash
                    ):
                        vectors = np.asarray(saved["embeddings"], dtype=np.float32)
                        if vectors.shape[0] != len(documents):
                            vectors = None
            except (OSError, ValueError, KeyError):
                vectors = None
        if vectors is None:
            vectors = self._encode(documents)
            np.savez_compressed(
                cache_path,
                model=np.asarray(self.model_name),
                content_hash=np.asarray(content_hash),
                embeddings=vectors,
            )
        self._matrix = vectors
        if self.backend == "faiss":
            try:
                faiss = importlib.import_module("faiss")
            except ImportError as exc:
                raise RuntimeError(
                    "FAISS backend requested but FAISS is not installed."
                ) from exc
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._index = index
            self.used_faiss = True
        else:
            self._index = None
            self.used_faiss = False
        return self

    def rank(self, query: str, *, limit: int | None = None) -> list[tuple[int, float]]:
        """Return documents ranked by cosine similarity to the query."""
        if self._matrix is None:
            raise RuntimeError("Call fit() before rank().")
        query_vector = self._encode([query])
        amount = len(self._matrix) if limit is None else min(limit, len(self._matrix))
        if self._index is not None:
            distances, indices = self._index.search(query_vector, amount)
            return [
                (int(index), float(score))
                for index, score in zip(indices[0], distances[0])
                if index >= 0
            ]
        scores = self._matrix @ query_vector[0]
        indices = np.argsort(-scores, kind="stable")[:amount]
        return [(int(index), float(scores[index])) for index in indices]


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL_ENV",
    "DenseRetriever",
    "configured_model_name",
]
