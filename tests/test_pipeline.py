import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RetrievalPipeline
from tests.fakes import fake_embed, fake_rerank_score

V1_SNIPPETS = {
    "s1": "function normalize(str) { const s2 = str.trim(); return forward(s2); }",
    "s2": "function check(s) { var pre = s.slice(0,6); return pre === 'en-US'; }",
    "s3": "function perf(str) { if (act(A, str)) { return act(B, str) } }",
}

# v2: s1 gets a tiny tweak (near-duplicate), s2 unchanged, s4 is new
V2_SNIPPETS = {
    "s1": "function normalize(str) { const s2 = str.trim().toLowerCase(); return forward(s2); }",
    "s2": "function check(s) { var pre = s.slice(0,6); return pre === 'en-US'; }",
    "s3": "function perf(str) { if (act(A, str)) { return act(B, str) } }",
    "s4": "function preprocessInput(raw) { return normalize(raw.trim()); }",
}


def build_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(embed_fn=fake_embed, rerank_score_fn=fake_rerank_score)


def test_index_v1_marks_everything_added():
    pipe = build_pipeline()
    stats = pipe.index_corpus_version("v1", V1_SNIPPETS)
    assert stats.total == 3
    assert stats.added == 3
    assert stats.unchanged == 0
    assert stats.changed == 0


def test_index_v2_only_reembeds_changed_and_new():
    pipe = build_pipeline()
    pipe.index_corpus_version("v1", V1_SNIPPETS)
    stats = pipe.index_corpus_version("v2", V2_SNIPPETS)
    assert stats.total == 4
    assert stats.unchanged == 2  # s2 and s3 identical
    assert stats.changed == 1  # s1 tweaked
    assert stats.added == 1  # s4 new
    assert set(pipe.index.id_to_text.keys()) == {"s1", "s2", "s3", "s4"}


def test_near_duplicate_detection_across_versions():
    pipe = build_pipeline()
    pipe.index_corpus_version("v1", V1_SNIPPETS)
    pipe.index_corpus_version("v2", V2_SNIPPETS)
    pairs = pipe.index.find_near_duplicates(similarity_threshold=0.8)
    flagged_ids = {p[0] for p in pairs}
    assert "s1" in flagged_ids  # tiny tweak should be flagged as near-duplicate
    assert "s2" not in flagged_ids  # unchanged, not a "changed" candidate at all


def test_retrieve_ranks_relevant_snippet_first():
    pipe = build_pipeline()
    pipe.index_corpus_version("v1", V1_SNIPPETS)
    results = pipe.retrieve("how is input preprocessed before main function", final_k=3)
    assert len(results) > 0
    top_id = results[0][0]
    # s1 (normalize) is the actually-relevant snippet for this query.
    assert top_id == "s1"


def test_retrieve_without_reranker_still_returns_shortlist():
    pipe = RetrievalPipeline(embed_fn=fake_embed)  # no reranker
    pipe.index_corpus_version("v1", V1_SNIPPETS)
    results = pipe.retrieve("input preprocessing normalize", final_k=3)
    assert len(results) > 0
