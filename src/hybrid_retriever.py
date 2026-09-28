"""Fuse dense + sparse candidate lists into one shortlist.

Uses Reciprocal Rank Fusion (RRF) by default — it's parameter-light and robust
across score scales (BM25 and cosine similarity aren't directly comparable, so
naive weighted-sum fusion needs careful tuning; RRF sidesteps that).
"""
from __future__ import annotations

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))
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
        intent: str = "behavior",
    ) -> list[tuple[str, float]]:
        """`query_variants` lets you pass the cleaned query plus any
        sub-queries/expansions from QueryProcessor.
        Dynamic weights adjust dense vs. sparse emphasis depending on query intent.
        """
        ranked_lists = []
        weights = []

        # Intent-aware dynamic weights
        sparse_w = 1.5 if intent in ("definition", "usage") else 1.0
        dense_w = 1.2 if intent == "behavior" else 1.0

        for idx, q in enumerate(query_variants):
            # Main query variant gets full weight, expanded queries get 0.8 weight
            variant_w = 1.0 if idx == 0 else 0.8

            dense_res = self.dense.search(q, top_k=top_k_each)
            if dense_res:
                ranked_lists.append(dense_res)
                weights.append(dense_w * variant_w)

            sparse_res = self.sparse.search(q, top_k=top_k_each)
            if sparse_res:
                ranked_lists.append(sparse_res)
                weights.append(sparse_w * variant_w)

        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
        return fused[:shortlist_size]
