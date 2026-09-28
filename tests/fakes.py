"""A hash-based fake embedder so the pipeline is testable without network
access or a real model. It's *not* semantically meaningful — it's just
deterministic and consistent, enough to verify the wiring (indexing, hybrid
fusion, rerank, incremental updates) actually works end-to-end.
"""
from __future__ import annotations

import hashlib

import numpy as np

_DIM = 32


def fake_embed(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    vecs = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in text.lower().split():
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            vecs[i, h % _DIM] += 1.0
    return vecs


def fake_rerank_score(query: str, doc_text: str) -> float:
    """Token-overlap score standing in for a cross-encoder."""
    q_tokens = set(query.lower().split())
    d_tokens = set(doc_text.lower().split())
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)
