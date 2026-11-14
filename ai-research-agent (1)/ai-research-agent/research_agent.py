"""Agentic literature-review pipeline.

    Strategist (LLM agent)  -> title, keywords, search queries
    Retrieval  (code)       -> arXiv / OpenAlex / Semantic Scholar, hard date gate (>= 2024)
    Analyst    (LLM agent)  -> per-paper notes + thematic clusters from abstracts
    Writer     (LLM agent)  -> Introduction, Synthesis, Discussion, Future work, Abstract, Conclusion
    Verifier   (code)       -> deletes citations that were not retrieved, renumbers the rest
    Publisher  (code)       -> figures from real data + IEEE-style PDF, Markdown, BibTeX, ZIP bundle

Retrieval is deliberately *not* an LLM tool call: the model never sees a URL it could mangle,
and every reference in the final paper comes from structured API metadata.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import figures as fig_mod
import llm_client
from llm_client import TokenBudget, clean_api_key
from paper_builder import PaperDraft, assemble_paper, bibtex, to_markdown
from parsing import Analysis, Plan, parse_analysis, parse_markers, parse_plan
from pdf_builder import build_pdf
from sources import MIN_YEAR, Paper, retrieve

run_agent = llm_client.run_agent  # module attribute so tests can replace it

RULES = (
    "Rules:\n"
    "- Cite evidence ONLY with bracketed ids that appear in the notes, for example [3] or [2, 5]. "
    "Never cite any other number. Never invent papers, authors, numbers, datasets or results.\n"
    "- Use only information contained in the notes. If the evidence is thin, say so.\n"
    "- Formal academic prose, third person. Paraphrase; do not copy.\n"
    "- Plain text only. No bold, no bullet lists, no tables, no markdown headings unless explicitly requested."
)


class NotEnoughPapers(RuntimeError):
    pass


@dataclass
class ResearchResult:
    draft: PaperDraft
    papers: list
    stats: dict
    warnings: list
    figures: dict
    pdf: bytes
    markdown: str
    bibtex: str
    bundle: bytes
    plan: Plan
    analysis: Analysis
    queries: list = field(default_factory=list)


def _short(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def _notes_block(papers: list, analysis: Analysis) -> str:
    lines = []
    for p in papers:
        c, m, l = analysis.notes[p.id]
        lines.append(f"[{p.id}] {_short(p.title, 14)} ({p.published:%Y-%m}). Contribution: {c} Method: {m} Limitation: {l}")
    themes = [f"- {name} (studies {', '.join(map(str, ids))}): {summary}" for name, ids, summary in analysis.themes]
    return "STUDY NOTES:\n" + "\n".join(lines) + "\n\nTHEMES:\n" + "\n".join(themes)


def generate_paper(
    topic: str,
    *,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
    n_papers: int = 12,
    min_year: int = MIN_YEAR,
    tokens_per_minute: int = 7000,
    author_line: str = "",
    openalex_key: str = "",
    s2_key: str = "",
    use_semantic_scholar: bool = True,
    progress: Optional[Callable[[str, float], None]] = None,
) -> ResearchResult:
    """Run the full pipeline and return every artefact of the report."""
    say = progress or (lambda _msg, _frac: None)
    api_key = clean_api_key(api_key)                      # raises ApiKeyError with a clear diagnosis
    if min_year < MIN_YEAR:
        raise ValueError(f"Earliest allowed year is {MIN_YEAR}.")
    topic = " ".join(topic.split())
    today = date.today()
    start = date(min_year, 1, 1)

    budget = TokenBudget(tokens_per_minute, on_wait=lambda s: say(f"Pacing requests for the Groq token limit ({s:.0f}s)...", -1))
    status = lambda msg: say(msg, -1)  # noqa: E731 - -1 means "keep the current progress value"

    def agent(role, goal, backstory, prompt, expected, out_tokens=1200, max_tokens=2500) -> str:
        return run_agent(model=model, api_key=api_key, role=role, goal=goal, backstory=backstory,
                         prompt=prompt, expected_output=expected, budget=budget,
                         expected_out_tokens=out_tokens, max_tokens=max_tokens, on_status=status)

    # ---------------------------------------------------------------- 1. plan
    say("Agent 1/3 (Research Strategist): planning the search...", 0.04)
    plan_text = agent(
        "Research Strategist",
        "Turn a research topic into a precise search plan for an academic literature review.",
        "You are an expert research librarian who designs effective database queries.",
        f"Research topic: {topic}\n"
        f"Only literature published since 1 January {min_year} will be searched.\n\n"
        "Produce a search plan using EXACTLY this line format and nothing else:\n"
        "TITLE: <concise academic paper title, at most 14 words>\n"
        "KEYWORDS: <8 to 10 keyword phrases separated by semicolons>\n"
        "QUERY: <search query of 2 to 5 words covering the core concept>\n"
        "QUERY: <search query covering methods or techniques>\n"
        "QUERY: <search query covering applications>\n"
        "QUERY: <search query covering challenges, evaluation or limitations>",
        "Lines starting with TITLE:, KEYWORDS: and four QUERY: lines.",
        out_tokens=350,
    )
    plan = parse_plan(plan_text, topic)

    # ---------------------------------------------------------------- 2. retrieve
    say("Retrieving papers published since " + start.strftime("%B %Y") + "...", 0.12)
    per_query = max(10, int(n_papers * 1.4))
    papers, stats, warnings = retrieve(
        plan.queries, plan.keywords, start=start, end=today, per_query=per_query, n_final=n_papers,
        openalex_key=openalex_key, s2_key=s2_key, use_semantic_scholar=use_semantic_scholar,
        progress=lambda m: say(m, -1),
    )
    if len(papers) < 3:
        raise NotEnoughPapers(
            f"Only {len(papers)} usable paper(s) were found for this topic since {start:%B %Y}. "
            "Try a broader topic or add an OpenAlex API key. Details: " + " | ".join(warnings[:3]))

    # ---------------------------------------------------------------- 3. analyse
    say(f"Agent 2/3 (Literature Analyst): reading {len(papers)} abstracts...", 0.30)
    listing = "\n".join(
        f"ID {p.id} | {_short(p.title, 18)} ({p.published:%Y-%m}) | {_short(p.abstract, 70)}" for p in papers)
    analysis_text = agent(
        "Literature Analyst",
        "Extract what each paper contributes and group the papers into coherent themes.",
        "You are a meticulous postdoctoral researcher. You summarise only what an abstract actually states.",
        f"Research topic: {topic}\n\nPAPERS (title, date, abstract excerpt):\n{listing}\n\n"
        "For EVERY paper write one line in exactly this format:\n"
        "PAPER <id> | C: <main contribution, max 20 words> | M: <method or data, max 12 words> | "
        "L: <limitation or caveat stated, max 12 words, or not stated>\n\n"
        "Then write 3 to 5 theme lines, each paper appearing in at least one theme:\n"
        "THEME: <theme name, 2 to 5 words> | IDS: <comma-separated paper ids> | S: <one sentence, max 25 words>\n\n"
        "Output only these lines.",
        "One PAPER line per paper followed by 3 to 5 THEME lines.",
        out_tokens=1500, max_tokens=3500,
    )
    analysis = parse_analysis(analysis_text, papers)
    if analysis.used_fallback:
        warnings.append("The analyst answer was incomplete; some notes were taken directly from abstracts.")

    # ---------------------------------------------------------------- 4. write
    notes = _notes_block(papers, analysis)
    n = len(papers)
    window = f"{start:%B %Y} to {today:%B %Y}"
    writer = dict(
        role="Academic Writer",
        goal="Write rigorous, well-structured sections of a literature-review paper grounded in the supplied notes.",
        backstory="You are a senior researcher who writes clearly, cites carefully and never overstates evidence.",
    )
    written: dict = {}

    say("Agent 3/3 (Academic Writer): drafting the Introduction...", 0.45)
    written["introduction"] = agent(
        **writer,
        prompt=(
            f'Write the INTRODUCTION of a literature-review paper titled "{plan.title}" on: {topic}.\n'
            f"The paper reviews {n} studies published {window}.\n\n{notes}\n\n{RULES}\n\n"
            "Write 3 paragraphs, about 230 words in total: (1) background and motivation; (2) the open problem and why "
            f"work since {min_year} matters; (3) the aim of this review and an outline (Section II methodology, III overview "
            "of studies, IV thematic synthesis, V discussion, VI future directions, VII conclusion).\n"
            "Output only the paragraphs separated by blank lines."),
        expected="Three paragraphs of academic prose.", out_tokens=700)

    say("Agent 3/3 (Academic Writer): writing the thematic synthesis...", 0.58)
    theme_names = "\n".join(f"- {name}" for name, _ids, _s in analysis.themes)
    written["synthesis"] = agent(
        **writer,
        prompt=(
            f"Write the THEMATIC SYNTHESIS of the paper on: {topic}.\n\n{notes}\n\n{RULES}\n\n"
            "For each theme listed below write a subsection: a line starting with '### ' and the theme name, then ONE "
            "paragraph of about 110 words that compares the studies in that theme, cites each with [id], and points out "
            "agreement, disagreement or differences in method.\nThemes:\n" + theme_names),
        expected="One '### theme' heading plus one paragraph per theme.", out_tokens=1400, max_tokens=3500)

    say("Agent 3/3 (Academic Writer): writing discussion and future directions...", 0.72)
    parts = parse_markers(agent(
        **writer,
        prompt=(
            f"Write two sections of the paper on: {topic}.\n\n{notes}\n\n{RULES}\n\n"
            "Output exactly two parts, each starting with its marker on its own line:\n"
            "@@DISCUSSION\ntwo paragraphs (about 170 words): challenges and limitations visible in the evidence, with citations.\n"
            "@@FUTURE\ntwo paragraphs (about 170 words): research gaps and concrete future research directions that follow "
            "from those limitations, with citations where applicable."),
        expected="Two parts introduced by @@DISCUSSION and @@FUTURE.", out_tokens=900))
    written["discussion"] = parts.get("discussion", "")
    written["future"] = parts.get("future", "")

    say("Agent 3/3 (Academic Writer): writing abstract and conclusion...", 0.84)
    src_names = ", ".join(stats["raw_by_source"]) or "scholarly databases"
    parts = parse_markers(agent(
        **writer,
        prompt=(
            f'Paper title: "{plan.title}". Topic: {topic}.\n'
            f"Method facts: {n} studies from {src_names} dated {window} were included after screening "
            f"{stats['raw_total']} retrieved records; titles and abstracts were analysed.\n\n{notes}\n\n{RULES}\n\n"
            "Output exactly two parts, each starting with its marker on its own line:\n"
            "@@ABSTRACT\none paragraph of 150 to 180 words: background, aim, method, main findings per theme, implications. "
            "NO citations in the abstract.\n"
            "@@CONCLUSION\none paragraph of about 110 words summarising the findings and outlook; citations allowed."),
        expected="Two parts introduced by @@ABSTRACT and @@CONCLUSION.", out_tokens=700))
    written["abstract"] = parts.get("abstract", "")
    written["conclusion"] = parts.get("conclusion", "")
    if not written["abstract"]:
        written["abstract"] = analysis.themes[0][2] if analysis.themes else plan.title
        warnings.append("The abstract could not be parsed from the model answer; a fallback was used.")

    # ---------------------------------------------------------------- 5. verify + publish
    say("Verifying citations and building figures...", 0.92)
    draft = assemble_paper(topic=topic, plan=plan, papers=papers, analysis=analysis, written=written, stats=stats,
                           queries=plan.queries, start=start, end=today, model=model,
                           author_line=author_line, today=today)
    if draft.invalid_citations_removed:
        warnings.append(f"Removed {draft.invalid_citations_removed} citation marker(s) that did not match any retrieved paper.")

    figs = {
        "flow": fig_mod.figure_flow(stats, f"{min_year}\u2013{today.year}"),
        "timeline": fig_mod.figure_timeline(papers),
        "themes": fig_mod.figure_themes(analysis.themes),
    }
    say("Rendering the PDF...", 0.96)
    pdf = build_pdf(draft, figs)
    files = {k: f"figures/{k}.png" for k in figs}
    markdown = to_markdown(draft, files)
    bib = bibtex(draft.references)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("paper.pdf", pdf)
        zf.writestr("paper.md", markdown)
        zf.writestr("references.bib", bib)
        for key, png in figs.items():
            zf.writestr(f"figures/{key}.png", png)
    say("Done.", 1.0)
    return ResearchResult(draft=draft, papers=papers, stats=stats, warnings=warnings, figures=figs, pdf=pdf,
                          markdown=markdown, bibtex=bib, bundle=buf.getvalue(), plan=plan, analysis=analysis,
                          queries=plan.queries)
