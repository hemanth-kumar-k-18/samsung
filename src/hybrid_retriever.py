"""Fuse dense + sparse candidate lists into one shortlist.

Uses Reciprocal Rank Fusion (RRF) by default — it's parameter-light and robust
across score scales (BM25 and cosine similarity aren't directly comparable, so
naive weighted-sum fusion needs careful tuning; RRF sidesteps that).
"""
from __future__ import annotations

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse

    def search(
        self,
        query_variants: list[str],
        top_k_each: int = 50,
        shortlist_size: int = 100,
    ) -> list[tuple[str, float]]:
        """`query_variants` lets you pass the cleaned query plus any
        sub-queries/expansions from QueryProcessor — each contributes its own
        ranked lists into the fusion, so decomposition/expansion actually pays
        off downstream instead of just being discarded."""
        ranked_lists = []
        for q in query_variants:
            ranked_lists.append(self.dense.search(q, top_k=top_k_each))
            ranked_lists.append(self.sparse.search(q, top_k=top_k_each))
        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:shortlist_size]
