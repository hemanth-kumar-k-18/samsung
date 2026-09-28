"""Version-aware incremental indexing.

P1 requirement: rebuild indexes/caches for a new codebase version in
reasonable time, not from scratch every time.

Approach: content-hash every processed snippet. On a new version, diff hashes
against the previous version's snapshot -> only unchanged snippets are
skipped; only new/changed snippets get (re-)embedded and upserted into the
dense + sparse indexes.

Bonus (evolutionary retrieval): `find_near_duplicates` flags snippets that are
nearly-but-not-exactly identical across versions (common when only a few
lines changed), so a downstream ranker can either collapse them to one
canonical entry with version metadata, or keep them distinct but explicitly
penalize near-duplicates from ranking too close together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .dense_retriever import DenseRetriever
from .snippet_processing import ProcessedSnippet
from .sparse_retriever import SparseRetriever


@dataclass
class IndexStats:
    total: int
    unchanged: int
    changed: int
    added: int


@dataclass
class VersionRecord:
    version: str
    snippet_hashes: dict[str, str] = field(default_factory=dict)  # id -> content_hash
    snippet_texts: dict[str, str] = field(default_factory=dict)  # id -> cleaned text snapshot


class IncrementalIndex:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse
        self._history: list[VersionRecord] = []
        self._id_to_text: dict[str, str] = {}
        self._id_to_snippet: dict[str, ProcessedSnippet] = {}

    def index_version(self, version: str, snippets: list[ProcessedSnippet]) -> IndexStats:
        prev_hashes = self._history[-1].snippet_hashes if self._history else {}

        to_upsert_ids: list[str] = []
        to_upsert_texts: list[str] = []
        unchanged = changed = added = 0

        for snip in snippets:
            snip.version = version
            embed_text = f"{snip.summary}\n{snip.cleaned}"
            self._id_to_text[snip.snippet_id] = embed_text
            self._id_to_snippet[snip.snippet_id] = snip

            prev_hash = prev_hashes.get(snip.snippet_id)
            if prev_hash == snip.content_hash:
                unchanged += 1
                continue
            elif prev_hash is None:
                added += 1
            else:
                changed += 1
            to_upsert_ids.append(snip.snippet_id)
            to_upsert_texts.append(embed_text)

        if to_upsert_ids:
            self.dense.upsert(to_upsert_ids, to_upsert_texts)
            self.sparse.upsert(to_upsert_ids, to_upsert_texts)

        new_hashes = {s.snippet_id: s.content_hash for s in snippets}
        new_texts = {s.snippet_id: s.cleaned for s in snippets}
        self._history.append(
            VersionRecord(version=version, snippet_hashes=new_hashes, snippet_texts=new_texts)
        )

        return IndexStats(
            total=len(snippets), unchanged=unchanged, changed=changed, added=added
        )

    @property
    def id_to_text(self) -> dict[str, str]:
        return self._id_to_text

    def find_near_duplicates(self, similarity_threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Compare snippets across the two most recent versions and flag pairs
        that are near-identical (changed but barely) — the hard case for
        evolutionary retrieval, where ranking must distinguish versions that
        look almost the same. Returns (old_id, new_id, similarity)."""
        if len(self._history) < 2:
            return []
        prev, curr = self._history[-2], self._history[-1]
        pairs = []
        for new_id, new_hash in curr.snippet_hashes.items():
            old_hash = prev.snippet_hashes.get(new_id)
            if old_hash is None or old_hash == new_hash:
                continue  # unchanged, or brand-new in this version
            old_text = prev.snippet_texts.get(new_id)
            new_text = curr.snippet_texts.get(new_id)
            if old_text is None or new_text is None:
                continue
            ratio = SequenceMatcher(None, old_text, new_text).ratio()
            if ratio >= similarity_threshold:
                pairs.append((new_id, new_id, ratio))
        return pairs
