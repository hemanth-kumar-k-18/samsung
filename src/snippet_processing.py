"""Snippet-side preprocessing: chunking, noise stripping, identifier extraction.

Uses lightweight regex-based structural splitting so it has zero heavy dependencies.
If you have `tree-sitter` + a grammar available on your training machine, swap
`chunk_by_function` for an AST-based version — the interface stays the same.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_FUNC_DEF_RE = re.compile(
    r"^\s*(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\(|def\s+\w+|"
    r"(?:public|private|protected|static|\s)*\w[\w<>\[\], ]*\s+\w+\s*\()",
    re.MULTILINE,
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LICENSE_HEADER_RE = re.compile(r"^\s*(//|#|/\*).{0,80}(license|copyright)", re.IGNORECASE)


@dataclass
class ProcessedSnippet:
    snippet_id: str
    raw: str
    cleaned: str
    identifiers: list[str]
    summary: str
    content_hash: str
    version: str | None = None
    metadata: dict = field(default_factory=dict)


def strip_boilerplate(code: str) -> str:
    lines = code.splitlines()
    kept = [ln for ln in lines if not _LICENSE_HEADER_RE.match(ln)]
    return "\n".join(kept).strip()


def extract_identifiers(code: str) -> list[str]:
    tokens = _IDENTIFIER_RE.findall(code)
    keywords = {
        "function", "const", "let", "var", "return", "if", "else", "for",
        "while", "def", "class", "public", "private", "static", "void",
        "int", "string", "bool", "true", "false", "null", "None", "self",
    }
    out, seen = [], set()
    for t in tokens:
        if t in keywords or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def naive_summarize(code: str, identifiers: list[str]) -> str:
    """Deterministic, model-free 'summary': surfaces the likely function name and
    call targets so there's a short natural-language-ish string to embed alongside
    the raw code. Replace with an actual LLM/codeT5 summarizer for real accuracy
    gains — this keeps the pipeline runnable without a model or network access.
    """
    name_match = re.search(r"(?:function|def)\s+(\w+)", code)
    fn_name = name_match.group(1) if name_match else (identifiers[0] if identifiers else "snippet")
    calls = [i for i in identifiers if i != fn_name][:5]
    calls_str = ", ".join(calls) if calls else "no external calls detected"
    return f"function {fn_name}; references: {calls_str}"


def chunk_by_function(code: str) -> list[str]:
    """Split code into function-ish chunks using boundary regex. Falls back to
    returning the whole snippet as one chunk if no boundaries are found — safe
    default for already-small snippets (like the CoIR apps dataset, which is
    typically already snippet-sized)."""
    matches = list(_FUNC_DEF_RE.finditer(code))
    if len(matches) <= 1:
        return [code]
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        chunks.append(code[start:end].strip())
    return chunks


def content_hash(code: str) -> str:
    normalized = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def process_snippet(snippet_id: str, raw_code: str, version: str | None = None) -> ProcessedSnippet:
    cleaned = strip_boilerplate(raw_code)
    identifiers = extract_identifiers(cleaned)
    summary = naive_summarize(cleaned, identifiers)
    return ProcessedSnippet(
        snippet_id=snippet_id,
        raw=raw_code,
        cleaned=cleaned,
        identifiers=identifiers,
        summary=summary,
        content_hash=content_hash(cleaned),
        version=version,
    )
