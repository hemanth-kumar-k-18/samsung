"""Sparse lexical retrieval over identifiers + tokenized code (BM25).

Strong complement to dense retrieval: exact identifier/function-name matches
(`normalize`, `forward`) are common in code queries and dense embeddings can
under-weight them relative to overall semantic similarity.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize_code(text: str) -> list[str]:
    """Split on non-identifier chars, then further split camelCase / snake_case
    so `normalizeInput` and `normalize input` both surface the same tokens."""
    raw_tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in raw_tokens:
        parts = re.sub(r"(?<!^)(?=[A-Z])", " ", tok).split()
        parts = [p for sub in parts for p in sub.split("_")]
        out.extend(p.lower() for p in parts if p)
    return out


class SparseRetriever:
    def __init__(self):
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    def build(self, ids: list[str], texts: list[str]) -> None:
        self._ids = list(ids)
        self._corpus_tokens = [tokenize_code(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def upsert(self, ids: list[str], texts: list[str]) -> None:
        """BM25 has no true incremental update, so we rebuild — but only over
        the (small) in-memory token lists, which is cheap compared to
        re-embedding with a neural model. Swap in a persistent inverted index
        (e.g. Tantivy/Whoosh) for large-scale incremental use."""
        id_to_row = {sid: i for i, sid in enumerate(self._ids)}
        for sid, text in zip(ids, texts):
            toks = tokenize_code(text)
            if sid in id_to_row:
                self._corpus_tokens[id_to_row[sid]] = toks
            else:
                self._ids.append(sid)
                self._corpus_tokens.append(toks)
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_code(query))
        ranked = sorted(zip(self._ids, scores), key=lambda x: -x[1])[:top_k]
        return [(sid, float(score)) for sid, score in ranked]
