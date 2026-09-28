"""Dense (embedding-based) retrieval.

Deliberately decoupled from any specific model: `embed_fn` is any callable
`list[str] -> np.ndarray[N, D]`. In production, pass a `sentence-transformers`
model's `.encode`. In tests, pass the deterministic fake in `tests/fakes.py`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]


class DenseRetriever:
    def __init__(self, embed_fn: EmbedFn):
        self.embed_fn = embed_fn
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def build(self, ids: list[str], texts: list[str]) -> None:
        self._ids = list(ids)
        vecs = self.embed_fn(texts)
        self._matrix = _normalize(np.asarray(vecs, dtype=np.float32))

    def upsert(self, ids: list[str], texts: list[str]) -> None:
        """Add or replace vectors for the given ids without re-embedding the
        whole corpus. Used by IncrementalIndex for fast version updates."""
        if self._matrix is None:
            self.build(ids, texts)
            return
        new_vecs = _normalize(np.asarray(self.embed_fn(texts), dtype=np.float32))
        id_to_row = {sid: i for i, sid in enumerate(self._ids)}
        for sid, vec in zip(ids, new_vecs):
            if sid in id_to_row:
                self._matrix[id_to_row[sid]] = vec
            else:
                self._ids.append(sid)
                self._matrix = np.vstack([self._matrix, vec[None, :]])

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self._matrix is None or not self._ids:
            return []
        q = _normalize(np.asarray(self.embed_fn([query]), dtype=np.float32))[0]
        scores = self._matrix @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]


def _normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
