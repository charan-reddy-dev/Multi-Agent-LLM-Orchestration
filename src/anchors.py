"""Deterministic anchor extraction and matching.

Anchors are extracted with rules, not by a model, so the same query always yields the same immutable
set and no LLM can invent, drop or rewrite one. Three kinds are recognised:

* numbers   – figures with optional currency and unit: "36%", "2.5 s", "$4.2M", "10,000 RPS", "Q3 2025"
* entities  – proper nouns, product names, acronyms and quoted terms: "General Motors", "PostgreSQL", "p99"
* constraints – machine-checkable instructions: "under 120 words", "exactly 3 bullet points",
                "do not mention Kafka"
"""

from __future__ import annotations

import re
from typing import Iterable

from src.schemas import Anchor, AnchorKind

UNITS = (
    r"%|percent|x|×|ms|s|sec|seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|"
    r"[KMBT]|k|bn|GB|TB|MB|KB|GiB|TiB|RPS|rps|req/s|QPS|qps|tokens?|users?|nodes?|replicas?|cores?|"
    r"engineers?|employees?|customers?|regions?|vCPUs?|GPUs?|USD|EUR|INR|bps|Gbps|Mbps|ops/s|"
    r"[a-z]{3,}s"  # any plural noun: "2 failures", "9 clusters", "6 workers"
)
NUMBER_RE = re.compile(
    rf"(?<![\w.\-/])(?:Q[1-4]\s)?[$€£₹]?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?:\s?(?:{UNITS})(?![\w]))?"
    rf"|(?<![\w.\-/])(?:Q[1-4]\s)?[$€£₹]?\d+(?:\.\d+)?(?:\s?(?:{UNITS})(?![\w]))?"
)

CONSTRAINT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("max_words", re.compile(r"\b(?:in\s+)?(?:under|at most|no more than|within|fewer than|maximum of|max(?:imum)?)\s+(\d+)\s+words\b", re.I)),
    ("bullets", re.compile(r"\b(?:exactly\s+|in\s+)?(\d+|two|three|four|five|six)\s+bullet(?:\s+point)?s\b", re.I)),
    ("forbid", re.compile(
        r"\b(?:do not|don't|must not|never|without)\s+(?:mention(?:ing)?|us(?:e|ing)|referenc(?:e|ing)|nam(?:e|ing))\s+"
        r"(\"[^\"]+\"|[A-Z][\w.+\-]*(?:\s+[A-Z][\w.+\-]*)*)"
    )),
]
WORD_NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

# Capitalised words that start instructions rather than name things.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "for", "in", "on", "of", "to", "with", "by", "at", "from",
    "as", "is", "are", "be", "this", "that", "these", "those", "it", "its", "our", "we", "you", "your", "they",
    "their", "each", "both", "all", "any", "what", "how", "why", "when", "which", "who", "where", "should", "would",
    "could", "can", "will", "must", "do", "does", "did", "not", "no", "yes", "please", "also", "only", "using",
    "use", "based", "given", "after", "before", "during", "while", "between", "across", "into", "per", "via",
    "compare", "explain", "summarize", "summarise", "write", "produce", "list", "describe", "draft", "recommend",
    "analyze", "analyse", "assess", "evaluate", "include", "keep", "return", "provide", "create", "prepare",
    "report", "identify", "estimate", "calculate", "outline", "mention", "avoid", "state", "cover", "review",
    "propose", "design", "plan", "decide", "choose", "justify", "highlight", "note", "consider", "first", "second",
    "third", "finally", "next", "then", "structure", "build", "migrate", "reduce", "improve", "assume", "suppose",
    "give", "tell", "show", "find", "make", "set", "get", "run", "help", "let", "shall", "add", "check", "ensure",
    "answer", "respond", "reply", "discuss", "detail", "translate", "rewrite", "convert", "format", "rank", "sort",
    "group", "classify", "suggest", "predict", "forecast", "quantify", "argue", "contrast", "define", "illustrate",
    "present", "focus", "end", "start", "begin", "conclude", "mark", "flag", "name", "rate", "score",
    "q1", "q2", "q3", "q4",
}
ENTITY_RE = re.compile(
    r"\"([^\"]{2,60})\""  # quoted terms
    r"|\b([A-Z][A-Za-z0-9]*(?:[\-.][A-Za-z0-9]+)*(?:\s+(?:of\s+|for\s+|&\s+)?[A-Z][A-Za-z0-9]*(?:[\-.][A-Za-z0-9]+)*)*)\b"
    r"|\b([a-z]+[A-Z][A-Za-z0-9]*|[a-z]\d{2,3})\b"  # camelCase / p99-style metric names
)


def _spans_overlap(a: tuple[int, int], spans: Iterable[tuple[int, int]]) -> bool:
    return any(a[0] < e and s < a[1] for s, e in spans)


def extract_anchors(query: str) -> tuple[Anchor, ...]:
    anchors: list[Anchor] = []
    taken: list[tuple[int, int]] = []
    forbidden_terms: set[str] = set()

    for rule_name, pattern in CONSTRAINT_PATTERNS:
        for m in pattern.finditer(query):
            raw = m.group(1).strip('"').rstrip(".,;:!?")
            value = int(raw) if raw.isdigit() else WORD_NUMBERS.get(raw.lower(), raw)
            if rule_name == "forbid":
                forbidden_terms.add(raw.lower())
            text = m.group(0).strip().rstrip(".,;:!?")
            anchors.append(Anchor(id="", kind=AnchorKind.CONSTRAINT, text=text, rule=(rule_name, value)))
            taken.append(m.span())

    seen_numbers: set[str] = set()
    for m in NUMBER_RE.finditer(query):
        if _spans_overlap(m.span(), taken):
            continue
        text = m.group(0).strip()
        taken.append(m.span())
        if normalize(text) in seen_numbers:
            continue
        seen_numbers.add(normalize(text))
        anchors.append(Anchor(id="", kind=AnchorKind.NUMBER, text=text))

    seen_entities: set[str] = set()
    for m in ENTITY_RE.finditer(query):
        text = next(g for g in m.groups() if g)
        span = m.span()
        if _spans_overlap(span, taken):
            continue
        words = text.split()
        # Drop instruction verbs and articles at the front of a capitalised run ("Compare Redis" -> "Redis").
        while words and words[0].lower() in STOPWORDS and not m.group(1):
            words = words[1:]
        while words and words[-1].lower() in {"of", "for", "&"}:
            words = words[:-1]
        text = " ".join(words)
        if not text or text.lower() in STOPWORDS or len(text) < 2:
            continue
        if text.lower() in forbidden_terms or text.lower() in seen_entities:
            continue
        seen_entities.add(text.lower())
        anchors.append(Anchor(id="", kind=AnchorKind.ENTITY, text=text))

    ordered = sorted(anchors, key=lambda a: (["number", "entity", "constraint"].index(a.kind.value), query.find(a.text)))
    prefix = {AnchorKind.NUMBER: "N", AnchorKind.ENTITY: "E", AnchorKind.CONSTRAINT: "C"}
    counters = {k: 0 for k in prefix}
    result = []
    for a in ordered:
        counters[a.kind] += 1
        result.append(Anchor(id=f"{prefix[a.kind]}{counters[a.kind]}", kind=a.kind, text=a.text, rule=a.rule))
    return tuple(result)


# ----------------------------------------------------------------------------------- matching


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("×", "x")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def contains_anchor(text: str, anchor: Anchor) -> bool:
    """True if the anchor appears verbatim (after case, whitespace and thousands-separator normalisation)."""
    needle = normalize(anchor.text)
    hay = normalize(text)
    left = r"(?<![\w.])" if needle[:1].isalnum() else ""
    right = r"(?![\w])" if needle[-1:].isalnum() else ""
    return re.search(left + re.escape(needle) + right, hay) is not None


NUMERIC_CORE_RE = re.compile(r"(?<![\w.\-/])\d+(?:,\d{3})*(?:\.\d+)?")


def numeric_values(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUMERIC_CORE_RE.finditer(text)}


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def count_bullets(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", text))
