# Agentic Code Retrieval

A CPU-friendly hybrid retrieval pipeline for ranking code snippets against natural-language
queries, built for the CoIR `AppsRetrieval` MTEB task.

## Architecture

```
query ──▶ QueryProcessor ──▶ [rewritten/expanded query variants]
                                        │
                                        ▼
                          ┌── DenseRetriever (bi-encoder, embeddings)
                          │        top-K candidates
              query ─────►┤
                          └── SparseRetriever (BM25 over identifiers/tokens)
                                        │
                                        ▼
                             HybridRetriever (score fusion, top-N)
                                        │
                                        ▼
                              CrossEncoderReranker (precision pass)
                                        │
                                        ▼
                                final ranked list

corpus ──▶ SnippetProcessor (chunk/clean/summarize) ──▶ IncrementalIndex (versioned, diff-based)
```

- **`src/query_processing.py`** — cleans and expands the query, optionally decomposes
  multi-part questions, and classifies rough query intent (definition / behavior /
  usage / bugfix) used later to bias sparse-term weighting.
- **`src/snippet_processing.py`** — chunks raw code into semantically meaningful units
  (function/class level where possible), strips boilerplate/comments-only noise, and
  extracts identifier metadata (function names, calls, imports) for the sparse index.
- **`src/dense_retriever.py`** — wraps a `sentence-transformers` bi-encoder (swap in a
  code-specific model, e.g. a CodeBERT/UniXcoder/jina-code embedding model) for fast
  approximate nearest-neighbor retrieval over the whole corpus.
- **`src/sparse_retriever.py`** — BM25 over extracted identifiers + tokenized code,
  which is strong on exact-name queries (`normalize`, `forward`, etc.) that dense
  embeddings sometimes blur.
- **`src/hybrid_retriever.py`** — fuses dense + sparse scores (weighted sum or
  reciprocal-rank fusion) to get a candidate shortlist (~50-100 items).
- **`src/reranker.py`** — cross-encoder pass over just the shortlist for a precision
  boost. This is usually the single largest lever for NDCG@10 / MRR.
- **`src/incremental_index.py`** — content-hash based diffing so that re-indexing a new
  code version only re-embeds changed/added snippets (P1 requirement), instead of a
  full rebuild.
- **`src/encoder.py`** — `PrePostPipelineEncoder`, the class that plugs directly into
  MTEB's `AbsEncoder` interface so you can run the official evaluation harness.

## Why this design maps to the goals

| Goal | How this repo addresses it |
|---|---|
| P0 Accuracy | Hybrid retrieval (dense+sparse) + cross-encoder reranker |
| P1 Versioning | `IncrementalIndex` diffs by content hash, only re-embeds deltas |
| Bonus Evolutionary | `IncrementalIndex` keeps per-version records + a dedup/canonicalization hook (see `find_near_duplicates`) so near-identical snippets across versions can be collapsed or explicitly version-boosted |

## Setup

This sandbox cannot reach huggingface.co, so model downloads must happen on your own
machine / Colab / Kaggle where you actually run the submission. Everything here is
wired against a pluggable `embed_fn`, so swapping in a real model is a one-line change
(see `scripts/run_mteb_eval.py`).

```bash
pip install -r requirements.txt
```

Suggested starting models (all CPU-friendly, all on huggingface):
- Dense: `flax-sentence-embeddings/st-codesearch-distilroberta-base` or
  `Salesforce/codet5p-110m-embedding` or `jinaai/jina-embeddings-v2-base-code`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (general) or a code-tuned
  cross-encoder if you find one that fits your compute budget

## Running the pipeline (smoke test, no internet needed)

```bash
python -m pytest tests/ -v
```

This runs the full pipeline end-to-end using a deterministic hash-based fake embedder
(`tests/fakes.py`) so you can verify the wiring (chunking → indexing → hybrid retrieval →
rerank → incremental update) works before swapping in real models.

## Running the actual MTEB evaluation

```bash
python scripts/run_mteb_eval.py
```

This loads `AppsRetrieval`, wraps your real model in `PrePostPipelineEncoder`, runs
`mteb.evaluate`, and writes `appsretrieval_results.json` — the file you upload to your
GitHub release.

## Repo layout

```
src/
  query_processing.py
  snippet_processing.py
  dense_retriever.py
  sparse_retriever.py
  hybrid_retriever.py
  reranker.py
  incremental_index.py
  encoder.py
scripts/
  run_mteb_eval.py
  demo.py
tests/
  fakes.py
  test_pipeline.py
requirements.txt
```
