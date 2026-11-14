"""Parsing of LLM answers and citation verification.

The agents are asked for simple line formats (not JSON): they are far more reliable with
smaller models and cheaper in tokens. Everything here has a deterministic fallback so a
malformed answer degrades the report instead of crashing the app.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sources import STOPWORDS, clean_ws


# ------------------------------------------------------------------ planner
@dataclass
class Plan:
    title: str
    keywords: list
    queries: list


def _strip_md(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip().strip("\"'`\u201c\u201d").strip()


def parse_plan(text: str, topic: str) -> Plan:
    title, keywords, queries = "", [], []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("*-\u2022# ").strip()
        m = re.match(r"(?i)^\**(title|keywords?|query)\s*\d*\s*\**\s*:\s*(.+)$", line)
        if not m:
            continue
        tag, value = m.group(1).lower(), _strip_md(m.group(2))
        if tag == "title" and not title:
            title = value
        elif tag.startswith("keyword"):
            keywords += [k.strip().lower() for k in re.split(r"[;,]", value) if k.strip()]
        elif tag == "query" and value:
            queries.append(value)

    topic_words = [w for w in re.findall(r"[A-Za-z0-9\-]+", topic) if w.lower() not in STOPWORDS and len(w) > 2]
    if not title:
        title = clean_ws(topic).rstrip("?.") or "Literature Review"
    if not queries:
        queries = [clean_ws(topic)[:90]]
    if not keywords:
        keywords = [w.lower() for w in topic_words][:8]
    # the topic's own words always count as keywords, so screening cannot drop everything
    keywords += [w.lower() for w in topic_words[:4]]

    def uniq(items, limit):
        seen, out = set(), []
        for it in items:
            if it.lower() not in seen:
                seen.add(it.lower())
                out.append(it)
        return out[:limit]

    return Plan(title=title[:160], keywords=uniq(keywords, 14), queries=uniq(queries, 5))


# ------------------------------------------------------------------ analyst
@dataclass
class Analysis:
    notes: dict = field(default_factory=dict)    # id -> (contribution, method, limitation)
    themes: list = field(default_factory=list)   # (name, [ids], summary)
    used_fallback: bool = False


_PAPER_RE = re.compile(r"^\W*PAPER\s+(\d+)\s*\|\s*C\s*:\s*(.*?)\s*\|\s*M\s*:\s*(.*?)\s*\|\s*L\s*:\s*(.*?)\s*$", re.I)
_THEME_RE = re.compile(r"^\W*THEME\s*:\s*(.*?)\s*\|\s*IDS?\s*:\s*([^|]*)\|\s*S\s*:\s*(.*?)\s*$", re.I)


def _first_sentence(text: str, max_words: int = 24) -> str:
    sentence = re.split(r"(?<=[.!?])\s", clean_ws(text), maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def parse_analysis(text: str, papers: list) -> Analysis:
    ids = {p.id for p in papers}
    result = Analysis()
    for raw in (text or "").splitlines():
        line = _strip_md(raw)
        m = _PAPER_RE.match(line)
        if m and int(m.group(1)) in ids:
            result.notes[int(m.group(1))] = (_strip_md(m.group(2)), _strip_md(m.group(3)), _strip_md(m.group(4)))
            continue
        t = _THEME_RE.match(line)
        if t:
            theme_ids = sorted({int(i) for i in re.findall(r"\d+", t.group(2)) if int(i) in ids})
            if theme_ids:
                result.themes.append((_strip_md(t.group(1)), theme_ids, _strip_md(t.group(3))))

    # deterministic fallback for anything the model skipped
    for p in papers:
        if p.id not in result.notes:
            result.used_fallback = True
            result.notes[p.id] = (_first_sentence(p.abstract), "not extracted", "not extracted")
    if not result.themes:
        result.used_fallback = True
        result.themes = [("Included studies", [p.id for p in papers], "Studies identified by the search.")]
    return result


# ------------------------------------------------------------------ section blocks
def clean_heading(text: str) -> str:
    text = re.sub(r"[*_`#:]+", "", text).strip()
    text = re.sub(r"^(?:[A-Z]|\d+|[IVX]+)[.)]\s+", "", text)
    return text.strip()


def split_blocks(text: str) -> list:
    """Turn model text into blocks: ('p', text) | ('h', text) | ('li', text)."""
    blocks, buf = [], []

    def flush():
        if buf:
            blocks.append(("p", clean_ws(" ".join(buf))))
            buf.clear()

    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        h = re.match(r"^#{1,4}\s*(.+)$", line)
        if h:
            flush()
            if clean_heading(h.group(1)):
                blocks.append(("h", clean_heading(h.group(1))))
            continue
        li = re.match(r"^[-*\u2022]\s+(.+)$", line)
        if li:
            flush()
            blocks.append(("li", li.group(1).strip()))
            continue
        buf.append(line)
    flush()
    return blocks


_MARKER_RE = re.compile(r"(?im)^[ \t#*]*@@\s*([A-Z_]+)\s*[*:]*[ \t]*$")


def parse_markers(text: str) -> dict:
    """Split ``@@NAME`` delimited output into ``{name_lower: text}``."""
    parts = {}
    matches = list(_MARKER_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[m.group(1).lower()] = text[m.end():end].strip()
    return parts


# ------------------------------------------------------------------ citations
_CITE_RE = re.compile(
    r"\[\s*(\d+(?:\s*[\u2013\-]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[\u2013\-]\s*\d+)?)*)\s*\]"
)


class CitationMapper:
    """Validates ``[n]`` markers and renumbers them in order of first appearance.

    * ids that are not in the retrieved set are deleted (the model invented them);
    * numbers are rewritten IEEE-style so the reference list order matches the text.
    """

    def __init__(self, valid_ids):
        self.valid = set(valid_ids)
        self.order: dict = {}   # original id -> new number
        self.removed = 0

    @staticmethod
    def _expand(body: str) -> list:
        ids = []
        for part in re.split(r"[,;]", body):
            part = part.strip()
            rng = re.match(r"^(\d+)\s*[\u2013\-]\s*(\d+)$", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                ids += list(range(a, b + 1)) if 0 <= b - a <= 12 else [a, b]
            elif part.isdigit():
                ids.append(int(part))
        return ids

    def assign(self, original_id: int) -> int:
        if original_id not in self.order:
            self.order[original_id] = len(self.order) + 1
        return self.order[original_id]

    def rewrite(self, text: str) -> str:
        def repl(m):
            good = []
            for i in self._expand(m.group(1)):
                if i in self.valid:
                    good.append(self.assign(i))
                else:
                    self.removed += 1
            return "[" + ", ".join(str(n) for n in sorted(set(good))) + "]" if good else ""

        out = _CITE_RE.sub(repl, text or "")
        out = re.sub(r"[ \t]+([,.;:])", r"\1", out)
        return re.sub(r"[ \t]{2,}", " ", out).strip()

    @staticmethod
    def strip_all(text: str) -> str:
        out = _CITE_RE.sub("", text or "")
        out = re.sub(r"[ \t]+([,.;:])", r"\1", out)
        return re.sub(r"[ \t]{2,}", " ", out).strip()
