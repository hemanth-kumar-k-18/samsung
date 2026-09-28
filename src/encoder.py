"""MTEB-facing encoder.

IMPORTANT CAVEAT: MTEB's standard retrieval evaluation flow scores
`encode_queries(...)` against `encode_corpus(...)` via plain cosine similarity —
it does not natively run an arbitrary multi-stage pipeline (hybrid fusion +
cross-encoder rerank) as part of that scoring loop. So:

  - `PrePostPipelineEncoder` below plugs your dense embedding model into MTEB's
    `AbsEncoder` interface correctly, and applies query/snippet preprocessing
    (query expansion text is embedded, snippets are embedded as
    "summary + cleaned code") before encoding — this alone should already beat
    a naive embed-raw-text baseline, and it satisfies the "submit a JSON via
    MTEB" screening requirement as specified in the guidelines.
  - For the *actual* hybrid+rerank pipeline's real quality (which you'll want
    for your own self-evaluation and for the hands-on demo), use
    `scripts/manual_eval.py`, which runs `RetrievalPipeline` end-to-end and
    computes NDCG@10 / MRR directly — same metrics, computed by hand against
    the same dataset, but exercising your full pipeline rather than only the
    embedding step.
  - Check MTEB's current docs/source for your installed version for any
    custom-search hook (some versions support passing a model with its own
    `search()` method to bypass plain cosine scoring) — if available, wire
    `RetrievalPipeline.retrieve` in directly instead of using the embed-only
    path below.
"""
from __future__ import annotations

import numpy as np

from .query_processing import process_query
from .snippet_processing import process_snippet

try:
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.types import PromptType
except ImportError:  # mteb is only needed for scripts/run_mteb_eval.py
    AbsEncoder = object
    PromptType = None


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, embed_fn, batch_size: int = 64):
        """`embed_fn`: list[str] -> np.ndarray[N, D]. Wire in your real
        sentence-transformers model's `.encode` here, e.g.:

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code")
            encoder = PrePostPipelineEncoder(embed_fn=model.encode)
        """
        self.embed_fn = embed_fn
        self.batch_size = batch_size

    def encode(
        self,
        sentences: list[str],
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        is_query = prompt_type == PromptType.query
        prepped = [self._prep_query(s) if is_query else self._prep_corpus(s) for s in sentences]
        return np.asarray(self.embed_fn(prepped, batch_size=self.batch_size))

    @staticmethod
    def _prep_query(text: str) -> str:
        pq = process_query(text)
        # Fold expansion terms into the embedded string; cheap and model-free.
        extra = " ".join(pq.expansions)
        return f"{pq.cleaned} {extra}".strip()

    @staticmethod
    def _prep_corpus(text: str) -> str:
        snip = process_snippet(snippet_id="_", raw_code=text)
        return f"{snip.summary}\n{snip.cleaned}"
