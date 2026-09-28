"""Demo the full pipeline (query processing -> hybrid retrieval -> rerank)
against the exact example from the hackathon guidelines. Run with:

    python scripts/demo.py

Uses the deterministic offline fake embedder by default so it runs without
internet/model downloads. Swap `fake_embed`/`fake_rerank_score` for your real
model + cross-encoder to see actual results.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RetrievalPipeline
from tests.fakes import fake_embed, fake_rerank_score

SNIPPETS = {
    "Code#1": "function normalize(str) { const str2 = str.trim(); return forward(str2); }",
    "Code#2": "function check(s) { var pre = s.slice(0,6); return pre === 'en-US'; }",
    "Code#3": "function perf(str) { if (act(A, str)) { return act(B, str) } }",
}

DEFAULT_QUERY = "How is the input preprocessed before going to the main function?"


def run_query(pipe: RetrievalPipeline, query: str) -> None:
    t0 = time.perf_counter()
    results = pipe.retrieve(query, final_k=3)
    query_time = time.perf_counter() - t0

    print(f"\nQuery: {query}")
    print(f"Retrieved in {query_time*1000:.1f}ms\n")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"  {rank}. {doc_id}  (score={score:.4f})")
        print(f"     {SNIPPETS[doc_id]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo the Retrieval Pipeline")
    parser.add_argument("query", nargs="?", default=None, help="Custom query string")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive REPL mode")
    args = parser.parse_args()

    pipe = RetrievalPipeline(embed_fn=fake_embed, rerank_score_fn=fake_rerank_score)

    t0 = time.perf_counter()
    stats = pipe.index_corpus_version("v1", SNIPPETS)
    index_time = time.perf_counter() - t0
    print(f"Indexed {stats.total} snippets in {index_time*1000:.1f}ms "
          f"(added={stats.added}, changed={stats.changed}, unchanged={stats.unchanged})")

    if args.interactive:
        print("\n=== Interactive Retrieval REPL (type 'exit' or 'q' to quit) ===")
        while True:
            try:
                user_q = input("\nEnter query > ").strip()
                if not user_q or user_q.lower() in ("exit", "q", "quit"):
                    break
                run_query(pipe, user_q)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        query_to_run = args.query if args.query else DEFAULT_QUERY
        run_query(pipe, query_to_run)


if __name__ == "__main__":
    main()
