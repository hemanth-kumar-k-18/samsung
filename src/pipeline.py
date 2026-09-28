"""Orchestrates the full retrieve -> fuse -> rerank flow described in README.md."""
from __future__ import annotations

from .dense_retriever import DenseRetriever, EmbedFn
from .hybrid_retriever import HybridRetriever
from .incremental_index import IncrementalIndex, IndexStats
from .query_processing import process_query
from .reranker import CrossEncoderReranker, ScoreFn
from .snippet_processing import process_snippet
from .sparse_retriever import SparseRetriever


class RetrievalPipeline:
    def __init__(self, embed_fn: EmbedFn, rerank_score_fn: ScoreFn | None = None):
        self.dense = DenseRetriever(embed_fn)
        self.sparse = SparseRetriever()
        self.index = IncrementalIndex(self.dense, self.sparse)
        self.hybrid = HybridRetriever(self.dense, self.sparse)
        self.reranker = CrossEncoderReranker(rerank_score_fn) if rerank_score_fn else None

    def index_corpus_version(
        self, version: str, raw_snippets: dict[str, str]
    ) -> IndexStats:
        """`raw_snippets` maps snippet_id -> raw code string for this version."""
        processed = [
            process_snippet(sid, code, version=version) for sid, code in raw_snippets.items()
        ]
        return self.index.index_version(version, processed)

    def retrieve(self, query: str, shortlist_size: int = 100, final_k: int = 10):
        pq = process_query(query)
        shortlist = self.hybrid.search(
            pq.all_variants, top_k_each=50, shortlist_size=shortlist_size, intent=pq.intent
        )
        if self.reranker is None:
            return shortlist[:final_k]
        return self.reranker.rerank(
            pq.cleaned, shortlist, self.index.id_to_text, top_k=final_k
        )
