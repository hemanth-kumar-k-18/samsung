"""Precision reranking pass over a small shortlist (~50-100 candidates).

Decoupled from any specific model via `score_fn`: `(query, doc_text) -> float`.
In production, back this with a `sentence-transformers` CrossEncoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2` or a code-tuned equivalent). Because
it only runs on the shortlist, not the full corpus, this stays cheap even on
CPU — that's the whole point of the two-stage design.
"""
from __future__ import annotations

from typing import Callable

ScoreFn = Callable[[str, str], float]


class CrossEncoderReranker:
    def __init__(self, score_fn: ScoreFn):
        self.score_fn = score_fn

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        id_to_text: dict[str, str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        scored = [
            (doc_id, self.score_fn(query, id_to_text[doc_id]))
            for doc_id, _ in candidates
            if doc_id in id_to_text
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
