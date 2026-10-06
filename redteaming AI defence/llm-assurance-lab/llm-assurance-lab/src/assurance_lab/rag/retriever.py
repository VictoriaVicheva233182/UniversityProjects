"""Retrieval over document chunks.

The default retriever is TF-IDF because it is fast, deterministic and needs no
model download. To use dense embeddings (for example sentence-transformers with
FAISS), implement the ``Retriever`` protocol and pass it to the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from assurance_lab.rag.documents import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


ChunkFilter = Callable[[Chunk], bool]


class Retriever(Protocol):
    def search(self, query: str, k: int, allow: ChunkFilter | None = None) -> list[ScoredChunk]: ...


class TfidfRetriever:
    def __init__(self, chunks: Sequence[Chunk], min_score: float = 0.03) -> None:
        self.chunks = list(chunks)
        self.min_score = min_score
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1
        )
        self._matrix = self._vectorizer.fit_transform([c.search_text for c in self.chunks])

    def search(self, query: str, k: int, allow: ChunkFilter | None = None) -> list[ScoredChunk]:
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        order = np.argsort(-scores)
        results: list[ScoredChunk] = []
        for idx in order:
            score = float(scores[idx])
            if score < self.min_score:
                break
            chunk = self.chunks[idx]
            if allow is not None and not allow(chunk):
                continue
            results.append(ScoredChunk(chunk, score))
            if len(results) >= k:
                break
        return results
