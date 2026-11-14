"""Offline tests: no network, no real LLM. Run with:  python -m pytest -q"""
import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research_agent  # noqa: E402
import sources  # noqa: E402
from llm_client import ApiKeyError, TokenBudget, clean_api_key  # noqa: E402
from parsing import CitationMapper, parse_analysis, parse_markers, parse_plan, split_blocks  # noqa: E402
from sources import Paper  # noqa: E402

GOOD_KEY = "gsk_" + "A1b2C3d4E5" * 5 + "xy"  # 56 characters, like a real Groq key


# ------------------------------------------------------------- the user's actual bug
def test_reproduces_the_reported_error_position():
    bad = GOOD_KEY + "\u2026" * 32  # 32 stray non-ASCII characters after the key
    with pytest.raises(UnicodeEncodeError) as exc:
        ("Bearer " + bad).encode("ascii")
    assert (exc.value.start, exc.value.end) == (63, 95)  # "position 63-94" in the traceback


def test_clean_api_key_fixes_or_explains():
    assert clean_api_key(f'  "{GOOD_KEY}"  ') == GOOD_KEY                       # quotes / spaces
    assert clean_api_key(f"GROQ_API_KEY={GOOD_KEY}") == GOOD_KEY                # whole line pasted
    assert clean_api_key("\u200b" + GOOD_KEY + "\ufeff") == GOOD_KEY            # invisible characters
    assert clean_api_key(f"\u201c{GOOD_KEY}\u201d") == GOOD_KEY                 # curly quotes
    with pytest.raises(ApiKeyError, match="U\\+2026"):
        clean_api_key(GOOD_KEY + "\u2026\u2026")                                # real junk: explained
    with pytest.raises(ApiKeyError):
        clean_api_key("")


def test_token_budget_waits_and_expires(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("llm_client.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("llm_client.time.sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    waits = []
    b = TokenBudget(5000, on_wait=waits.append)
    b.acquire(3000)
    b.acquire(1500)
    assert not waits
    b.acquire(2000)            # would exceed 5000 -> must wait for the first event to expire
    assert waits and clock["t"] >= 1060


# ------------------------------------------------------------- parsing
def test_parse_plan_and_fallbacks():
    plan = parse_plan("TITLE: **Agentic AI for Science**\nKEYWORDS: agentic ai; llm agents, tool use\n"
                      "QUERY: agentic AI science\nQUERY 2: autonomous research agents", "agentic AI in science")
    assert plan.title == "Agentic AI for Science"
    assert "llm agents" in plan.keywords and len(plan.queries) == 2
    empty = parse_plan("garbage", "quantum error correction")
    assert empty.queries and empty.keywords


def _papers(n=5):
    return [Paper(title=f"Paper {i}", authors=["A B"], published=date(2025, 1, i), abstract="x" * 100, id=i)
            for i in range(1, n + 1)]


def test_parse_analysis_with_fallback():
    text = ("PAPER 1 | C: Does X | M: Method | L: none\nnoise\nPAPER 99 | C: ghost | M: m | L: l\n"
            "THEME: Core ideas | IDS: 1, 2, 99 | S: Summary here.")
    a = parse_analysis(text, _papers(3))
    assert a.notes[1][0] == "Does X" and 99 not in a.notes
    assert a.used_fallback and a.themes[0][1] == [1, 2]


def test_citation_mapper_renumbers_and_removes_invented():
    m = CitationMapper(range(1, 6))
    out = m.rewrite("First [4] then [2, 4] and a range [1\u20133] but fake [99] and [2024].")
    assert out == "First [1] then [1, 2] and a range [2, 3, 4] but fake and."
    assert m.removed == 2 and m.order == {4: 1, 2: 2, 1: 3, 3: 4}
    assert CitationMapper.strip_all("Abstract [1, 2] text.") == "Abstract text."


def test_split_blocks_and_markers():
    blocks = split_blocks("Para one\ncontinues.\n\n### A. Theme Name\nPara two.\n- item")
    assert blocks == [("p", "Para one continues."), ("h", "Theme Name"), ("p", "Para two."), ("li", "item")]
    parts = parse_markers("@@ABSTRACT\nabc\n\n**@@CONCLUSION**\ndef")
    assert parts == {"abstract": "abc", "conclusion": "def"}


# ------------------------------------------------------------- sources
ARXIV_XML = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2501.01234v2</id><published>2025-01-03T10:00:00Z</published>
<title>Agentic  AI
 for Science</title><summary>  We study agents.  </summary>
<author><name>Ada Lovelace</name></author><author><name>Jos\u00e9 \u00c1lvarez</name></author>
<arxiv:journal_ref>Nature Machine Intelligence 7 (2025)</arxiv:journal_ref></entry>
<entry><id>http://arxiv.org/api/errors#x</id><title>Error</title></entry></feed>"""


def test_parse_arxiv_and_query_has_date_window():
    papers = sources.parse_arxiv(ARXIV_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2501.01234" and p.title == "Agentic AI for Science" and p.venue.startswith("Nature")
    q = sources.build_arxiv_query("agentic AI for science", date(2024, 1, 1), date(2026, 11, 14))
    assert "submittedDate:[202401010000 TO 202611142359]" in q and "all:agentic" in q


def test_parse_openalex_and_semantic_scholar():
    oa = sources.parse_openalex({"results": [{
        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/abc", "display_name": "T",
        "publication_date": "2025-02-01", "cited_by_count": 7, "type": "article",
        "authorships": [{"author": {"display_name": "Grace Hopper"}}],
        "primary_location": {"source": {"display_name": "J. AI"}, "landing_page_url": "http://x"},
        "abstract_inverted_index": {"Hello": [0], "world": [1]}}]})
    assert oa[0].abstract == "Hello world" and oa[0].doi == "10.1/abc"
    s2 = sources.parse_semantic_scholar({"data": [{"title": "T", "abstract": "a", "year": 2025,
                                                   "publicationDate": None, "authors": [{"name": "X Y"}],
                                                   "externalIds": {"ArXiv": "2501.1"}, "citationCount": 3}]})
    assert s2[0].published.year == 2025 and s2[0].arxiv_id == "2501.1"


def _mk(title, d, abstract="agentic agents " * 12, **kw):
    return Paper(title=title, authors=["Ann Lee", "Bo Chen"], published=d, abstract=abstract, **kw)


def test_retrieve_enforces_date_dedupes_and_screens(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    batch = [
        _mk("Agentic systems for scientific discovery", date(2025, 3, 1), arxiv_id="2503.1", sources=["arXiv"]),
        _mk("Agentic systems for scientific discovery", date(2025, 3, 1), doi="10.1/x", venue="Nature", sources=["Semantic Scholar"]),
        _mk("OLD agentic paper from 2023", date(2023, 12, 31), sources=["arXiv"]),          # must be excluded
        _mk("No abstract agentic", date(2025, 4, 1), abstract="", sources=["arXiv"]),        # excluded
        _mk("Cooking recipes", date(2025, 5, 1), abstract="pasta sauce " * 20, sources=["arXiv"]),  # irrelevant
        _mk("Another agents study", date(2026, 6, 1), arxiv_id="2606.2", sources=["arXiv"]),
    ]
    monkeypatch.setattr(sources, "search_arxiv", lambda q, s, e, n: list(batch))
    monkeypatch.setattr(sources, "search_semantic_scholar", lambda *a, **k: [])
    papers, stats, warns = sources.retrieve(["agentic science"], ["agentic", "agents"], start=date(2024, 1, 1),
                                            end=date(2026, 11, 14), per_query=10, n_final=5)
    titles = [p.title for p in papers]
    assert all(p.published >= date(2024, 1, 1) for p in papers)
    assert "OLD agentic paper from 2023" not in titles and "Cooking recipes" not in titles
    assert stats["duplicates"] == 1 and stats["irrelevant"] == 1 and stats["included"] == 2
    merged = next(p for p in papers if p.title.startswith("Agentic systems"))
    assert merged.venue == "Nature" and merged.arxiv_id == "2503.1" and merged.doi == "10.1/x"
    assert [p.id for p in papers] == [1, 2]
    with pytest.raises(ValueError):
        sources.retrieve(["q"], ["k"], start=date(2023, 1, 1), end=date(2026, 1, 1), per_query=5, n_final=3)


# ------------------------------------------------------------- full pipeline with fakes
def fake_agent(*, role, prompt, **kw):
    if role == "Research Strategist":
        return ("TITLE: Agentic AI for Scientific Discovery: A Review\nKEYWORDS: agentic; agents; scientific discovery; llm\n"
                "QUERY: agentic AI science\nQUERY: LLM research agents\nQUERY: autonomous discovery\nQUERY: agent evaluation")
    if role == "Literature Analyst":
        ids = [int(i) for i in re.findall(r"^ID (\d+) \|", prompt, re.M)]
        lines = [f"PAPER {i} | C: Proposes method {i} for agents | M: LLM planning | L: small benchmark" for i in ids]
        half = len(ids) // 2
        lines += [f"THEME: Planning and tools | IDS: {', '.join(map(str, ids[:half + 1]))} | S: Studies on planning.",
                  f"THEME: Evaluation and safety | IDS: {', '.join(map(str, ids[half:]))} | S: Studies on evaluation."]
        return "\n".join(lines)
    if "INTRODUCTION" in prompt:
        return "Agents matter [3].\n\nRecent work [1, 2] differs [99].\n\nThis review follows [5\u20137]."
    if "THEMATIC SYNTHESIS" in prompt:
        return "### Planning and tools\nStudies [4] and [2] agree.\n\n### Evaluation and safety\nOthers [6] differ."
    if "@@DISCUSSION" in prompt:
        return "@@DISCUSSION\nChallenges persist [2].\n@@FUTURE\nFuture work should test [8] \u2014 \u03b1-agents."
    return "@@ABSTRACT\nThis review examines agentic AI [1] across themes.\n@@CONCLUSION\nAgents help [3]."


def test_full_pipeline_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(research_agent, "run_agent", fake_agent)
    corpus = [_mk(f"Agentic study number {i} on discovery", date(2024 + i % 3, 1 + i % 12, 5),
                  arxiv_id=f"24{i:02d}.0{i:04d}", sources=["arXiv"],
                  abstract=("agentic agents scientific discovery llm planning " * 8)) for i in range(1, 15)]
    corpus[0].authors = ["Jos\u00e9 \u00c1lvarez", "\u674e\u534e", "Grace Hopper", "Alan Turing"]  # Unicode names
    corpus[1].title = "Analysis of R&D <agents> & {LaTeX} $x_1$ costs in agentic discovery"
    monkeypatch.setattr(sources, "search_arxiv", lambda q, s, e, n: list(corpus))
    monkeypatch.setattr(sources, "search_semantic_scholar", lambda *a, **k: [])

    msgs = []
    res = research_agent.generate_paper("agentic AI for science", api_key=GOOD_KEY, n_papers=10,
                                        progress=lambda m, f: msgs.append((m, f)))
    assert msgs[-1] == ("Done.", 1.0)
    assert res.pdf.startswith(b"%PDF") and len(res.pdf) > 20_000
    assert res.draft.invalid_citations_removed >= 1                      # [99] was invented
    assert len(res.draft.references) == 10                               # all included studies are listed
    cited = {int(n) for grp in re.findall(r"\[(\d+(?:, \d+)*)\]", res.markdown) for n in grp.split(", ")}
    assert max(cited) <= 10
    assert "@article" in res.bibtex or "@misc" in res.bibtex
    assert res.bundle[:2] == b"PK"
    (tmp_path / "sample.pdf").write_bytes(res.pdf)
    (tmp_path / "sample.md").write_text(res.markdown, encoding="utf-8")


def test_not_enough_papers(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(research_agent, "run_agent", fake_agent)
    monkeypatch.setattr(sources, "search_arxiv", lambda q, s, e, n: [])
    monkeypatch.setattr(sources, "search_semantic_scholar", lambda *a, **k: [])
    with pytest.raises(research_agent.NotEnoughPapers):
        research_agent.generate_paper("anything", api_key=GOOD_KEY)
