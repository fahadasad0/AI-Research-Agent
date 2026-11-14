"""Scholarly retrieval: arXiv (no key), OpenAlex (free key), Semantic Scholar (optional key).

Every record is checked in *code* against the date window, so a paper published before
``MIN_YEAR`` can never reach the report, whatever a search backend returns.
"""
from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

import requests

MIN_YEAR = 2024
USER_AGENT = "ai-research-agent/2.0 (academic literature review tool)"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of",
    "on", "or", "the", "to", "using", "with", "via", "towards", "toward", "based", "study", "survey",
    "review", "approach", "new", "novel", "how", "what", "can", "do", "does",
}


@dataclass
class Paper:
    title: str
    authors: list
    published: date
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    citations: int = 0
    sources: list = field(default_factory=list)
    score: float = 0.0
    id: int = 0  # 1-based rank assigned after screening
    note: str = ""  # one-line contribution summary, filled by the analyst agent

    @property
    def year(self) -> int:
        return self.published.year

    @property
    def is_preprint_only(self) -> bool:
        v = self.venue.lower()
        return (not v) or "arxiv" in v or "preprint" in v

    def best_url(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return self.url


# ---------------------------------------------------------------- text helpers
def clean_ws(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stem_norm(text: str) -> str:
    """Lower-case, strip punctuation, crude plural stemming (agents -> agent)."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks)


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _tokens(query: str) -> list:
    return [t for t in re.findall(r"[A-Za-z0-9]+", query) if t.lower() not in STOPWORDS]


def _parse_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------- HTTP
def _get(url: str, params=None, headers=None, retries: int = 2, backoff: float = 5.0, timeout: int = 30):
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT, **(headers or {})})
        except requests.RequestException as exc:
            last = exc
            time.sleep(backoff * (attempt + 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            last = RuntimeError(f"HTTP {resp.status_code} from {url.split('/')[2]}")
            time.sleep(backoff * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} from {url.split('/')[2]}: {resp.text[:160]}")
        return resp
    raise last or RuntimeError("request failed")


# ---------------------------------------------------------------- arXiv
def build_arxiv_query(query: str, start: date, end: date) -> str:
    terms = _tokens(query)[:4] or [query.strip()]
    base = " AND ".join(f"all:{t}" for t in terms)
    window = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    return f"({base}) AND {window}"


def parse_arxiv(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        id_url = clean_ws(entry.findtext(f"{ATOM}id"))
        m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", id_url)
        published = _parse_date(entry.findtext(f"{ATOM}published"))
        if not m or not published:
            continue  # error entries / malformed records
        arxiv_id = m.group(1)
        authors = [clean_ws(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")]
        papers.append(Paper(
            title=clean_ws(entry.findtext(f"{ATOM}title")),
            authors=[a for a in authors if a],
            published=published,
            abstract=clean_ws(entry.findtext(f"{ATOM}summary")),
            venue=clean_ws(entry.findtext(f"{ARXIV_NS}journal_ref")) or "arXiv preprint",
            doi=clean_ws(entry.findtext(f"{ARXIV_NS}doi")),
            arxiv_id=arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            sources=["arXiv"],
        ))
    return papers


def search_arxiv(query: str, start: date, end: date, limit: int) -> list:
    resp = _get("https://export.arxiv.org/api/query", params={
        "search_query": build_arxiv_query(query, start, end),
        "start": 0, "max_results": limit, "sortBy": "relevance", "sortOrder": "descending",
    })
    return parse_arxiv(resp.text)


# ---------------------------------------------------------------- OpenAlex
def invert_abstract(index: Optional[dict]) -> str:
    if not index:
        return ""
    slots = {}
    for word, positions in index.items():
        for pos in positions:
            slots[pos] = word
    return clean_ws(" ".join(slots[i] for i in sorted(slots)))


def parse_openalex(payload: dict) -> list:
    papers = []
    for w in payload.get("results", []):
        if w.get("type") in {"paratext", "erratum", "retraction"}:
            continue
        published = _parse_date(w.get("publication_date"))
        if not published:
            continue
        loc = w.get("primary_location") or {}
        source = (loc.get("source") or {}).get("display_name") or ""
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        authors = [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", [])]
        papers.append(Paper(
            title=clean_ws(w.get("display_name") or w.get("title")),
            authors=[a for a in authors if a],
            published=published,
            abstract=invert_abstract(w.get("abstract_inverted_index")),
            venue=source,
            doi=doi,
            url=loc.get("landing_page_url") or w.get("id", ""),
            citations=int(w.get("cited_by_count") or 0),
            sources=["OpenAlex"],
        ))
    return papers


def search_openalex(query: str, start: date, end: date, limit: int, api_key: str) -> list:
    resp = _get("https://api.openalex.org/works", params={
        "search": query,
        "filter": f"from_publication_date:{start:%Y-%m-%d},to_publication_date:{end:%Y-%m-%d},has_abstract:true",
        "per-page": min(limit, 50),
        "api_key": api_key,
        "select": "id,doi,display_name,publication_date,authorships,primary_location,"
                  "abstract_inverted_index,cited_by_count,type",
    })
    return parse_openalex(resp.json())


# ---------------------------------------------------------------- Semantic Scholar
def parse_semantic_scholar(payload: dict) -> list:
    papers = []
    for w in payload.get("data") or []:
        published = _parse_date(w.get("publicationDate")) or (
            date(int(w["year"]), 7, 1) if w.get("year") else None)
        if not published:
            continue
        ext = w.get("externalIds") or {}
        papers.append(Paper(
            title=clean_ws(w.get("title")),
            authors=[a.get("name", "") for a in w.get("authors") or [] if a.get("name")],
            published=published,
            abstract=clean_ws(w.get("abstract")),
            venue=clean_ws(w.get("venue")),
            doi=ext.get("DOI", "") or "",
            arxiv_id=ext.get("ArXiv", "") or "",
            url=w.get("url", ""),
            citations=int(w.get("citationCount") or 0),
            sources=["Semantic Scholar"],
        ))
    return papers


def search_semantic_scholar(query: str, start: date, limit: int, api_key: str = "") -> list:
    resp = _get("https://api.semanticscholar.org/graph/v1/paper/search", params={
        "query": query, "year": f"{start.year}-", "limit": min(limit, 100),
        "fields": "title,abstract,authors,year,venue,externalIds,publicationDate,citationCount,url",
    }, headers={"x-api-key": api_key} if api_key else None, retries=1)
    return parse_semantic_scholar(resp.json())


# ---------------------------------------------------------------- merge / screen / rank
def merge_duplicates(papers: list) -> list:
    """Merge records describing the same work (same DOI, arXiv id or normalised title)."""
    index: dict = {}
    merged: list = []
    for p in papers:
        keys = [k for k in (
            f"doi:{p.doi.lower()}" if p.doi else "",
            f"arxiv:{p.arxiv_id.lower()}" if p.arxiv_id else "",
            f"title:{title_key(p.title)}" if len(title_key(p.title)) > 12 else "",
        ) if k]
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            merged.append(p)
            hit = p
        else:
            for attr in ("abstract", "venue", "doi", "arxiv_id", "url"):
                if not getattr(hit, attr) and getattr(p, attr):
                    setattr(hit, attr, getattr(p, attr))
            if hit.venue.lower().startswith("arxiv") and p.venue and "arxiv" not in p.venue.lower():
                hit.venue = p.venue
            hit.citations = max(hit.citations, p.citations)
            if len(p.authors) > len(hit.authors):
                hit.authors = p.authors
            hit.sources = sorted(set(hit.sources) | set(p.sources))
        for k in keys:
            index[k] = hit
    return merged


def relevance(p: Paper, keywords: list) -> tuple:
    title, abstract = stem_norm(p.title), stem_norm(p.abstract)
    t_hits = a_hits = 0
    for kw in keywords:
        k = f" {stem_norm(kw)} "
        if len(k.strip()) < 3:
            continue
        t_hits += f" {title} ".count(k) > 0
        a_hits += f" {abstract} ".count(k) > 0
    score = 2.0 * t_hits + 1.0 * a_hits + 0.25 * math.log1p(p.citations)
    return t_hits + a_hits, score


def retrieve(
    queries: list,
    keywords: list,
    *,
    start: date,
    end: date,
    per_query: int,
    n_final: int,
    openalex_key: str = "",
    s2_key: str = "",
    use_semantic_scholar: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple:
    """Search, date-gate, deduplicate, screen and rank. Returns ``(papers, stats, warnings)``."""
    if start.year < MIN_YEAR:
        raise ValueError(f"Start date must be on or after {MIN_YEAR}-01-01.")
    say = progress or (lambda _m: None)
    raw: list = []
    by_source: Counter = Counter()
    warnings: list = []

    def collect(name: str, fn, q: str):
        try:
            items = fn()
            raw.extend(items)
            by_source[name] += len(items)
        except Exception as exc:  # noqa: BLE001 - one failing backend must not stop the run
            warnings.append(f"{name} search failed for '{q}': {exc}")

    for i, q in enumerate(queries):
        say(f"Searching literature databases ({i + 1}/{len(queries)}): {q}")
        collect("arXiv", lambda: search_arxiv(q, start, end, per_query), q)
        if openalex_key:
            collect("OpenAlex", lambda: search_openalex(q, start, end, per_query, openalex_key), q)
        if use_semantic_scholar:
            collect("Semantic Scholar", lambda: search_semantic_scholar(q, start, per_query, s2_key), q)
        if i < len(queries) - 1:
            time.sleep(3.2)  # arXiv asks for roughly one request every 3 seconds
    if not openalex_key:
        warnings.append("OpenAlex skipped (it now needs a free API key; add OPENALEX_API_KEY to use it).")

    # hard date gate + must have title and a usable abstract
    valid = [p for p in raw if start <= p.published <= end and p.title and len(p.abstract) >= 80]
    unique = merge_duplicates(valid)

    scored = []
    for p in unique:
        hits, score = relevance(p, keywords)
        if hits > 0:
            p.score = score + (0.3 if p.year >= end.year else 0.0)
            scored.append(p)
    scored.sort(key=lambda p: (-p.score, -p.published.toordinal()))
    included = scored[:n_final]
    for rank, p in enumerate(included, start=1):
        p.id = rank

    assert all(p.published >= start for p in included), "date gate failed"  # belt and braces
    stats = {
        "raw_total": len(raw),
        "raw_by_source": dict(by_source),
        "dropped_date_or_abstract": len(raw) - len(valid),
        "duplicates": len(valid) - len(unique),
        "unique": len(unique),
        "irrelevant": len(unique) - len(scored),
        "screened": len(scored),
        "cutoff": len(scored) - len(included),
        "included": len(included),
    }
    return included, stats, warnings
