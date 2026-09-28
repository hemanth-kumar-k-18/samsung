"""Query-side preprocessing: cleaning, expansion, decomposition, intent tagging.

None of this requires a network call or a heavy model — it's cheap, deterministic
preprocessing that runs before any embedding happens. Swap in an LLM call for
`expand_query` if you want richer paraphrases; a heuristic version is provided so
the pipeline works fully offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "how", "what", "where",
    "when", "why", "does", "do", "did", "to", "of", "in", "on", "for",
    "and", "or", "before", "after", "it", "this", "that",
}

# Very small heuristic intent lexicon. Extend as you see more query patterns
# in the CoIR apps dataset.
_INTENT_KEYWORDS = {
    "definition": {"defined", "definition", "declare", "declared", "where is"},
    "behavior": {"how", "does", "work", "works", "logic", "algorithm"},
    "usage": {"call", "called", "use", "used", "invoke", "invoked"},
    "bugfix": {"bug", "error", "fail", "failing", "crash", "wrong", "incorrect"},
}


@dataclass
class ProcessedQuery:
    original: str
    cleaned: str
    tokens: list[str]
    intent: str
    sub_queries: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)

    @property
    def all_variants(self) -> list[str]:
        """Every string form worth embedding/searching for this query."""
        variants = [self.cleaned, *self.sub_queries, *self.expansions]
        # de-duplicate while preserving order
        seen, out = set(), []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out


def clean_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def tokenize(query: str) -> list[str]:
    return [t for t in clean_query(query).split() if t not in _STOPWORDS and len(t) > 1]


def classify_intent(query: str) -> str:
    q = query.lower()
    scores = {intent: sum(1 for kw in kws if kw in q) for intent, kws in _INTENT_KEYWORDS.items()}
    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0 else "behavior"


def decompose_query(query: str) -> list[str]:
    """Split a compound question into sub-questions on common connectors.

    e.g. "how is input validated and then passed to main" ->
         ["how is input validated", "then passed to main"]
    """
    parts = re.split(r"\band then\b|\bthen\b|\band\b|;", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else []


def expand_query(query: str, tokens: list[str]) -> list[str]:
    """Heuristic query expansion: map natural-language verbs to likely identifier
    fragments a developer would actually use in code (snake_case-ish tokens).

    This is intentionally simple and dependency-free. If you have LLM access at
    submission time, replace this with a prompt like:
        "Rewrite this question as short code-identifier-style phrases: {query}"
    and merge the results in.
    """
    synonym_map = {
        "preprocessed": ["preprocess", "normalize", "sanitize", "clean_input"],
        "input": ["input", "arg", "argument", "param"],
        "validated": ["validate", "check", "verify"],
        "main": ["main", "entrypoint", "run"],
        "output": ["output", "result", "return"],
        "error": ["error", "exception", "raise"],
        "loop": ["loop", "iterate", "for", "while"],
    }
    expansions: list[str] = []
    extra_terms: list[str] = []
    for tok in tokens:
        if tok in synonym_map:
            extra_terms.extend(synonym_map[tok])
    if extra_terms:
        expansions.append(" ".join(dict.fromkeys(extra_terms)))
    return expansions


def process_query(query: str) -> ProcessedQuery:
    cleaned = clean_query(query)
    tokens = tokenize(query)
    return ProcessedQuery(
        original=query,
        cleaned=cleaned,
        tokens=tokens,
        intent=classify_intent(query),
        sub_queries=decompose_query(query),
        expansions=expand_query(query, tokens),
    )
