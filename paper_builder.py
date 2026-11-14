"""Assemble the paper: sections, verified citations, IEEE-style references, Markdown, BibTeX.

Blocks are plain dicts so the PDF and Markdown writers can share them:
    {"type": "p", "text": ...}                      paragraph
    {"type": "h", "text": ...}                      sub-heading
    {"type": "li", "text": ...}                     bullet
    {"type": "fig", "key": ..., "caption": ...}     figure (PNG stored in the figures dict)
    {"type": "table", "caption", "header", "rows"}  table
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from parsing import Analysis, CitationMapper, Plan, split_blocks
from sources import Paper

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
MONTHS = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]


@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)


@dataclass
class PaperDraft:
    title: str
    author_line: str
    byline: str
    abstract: str
    keywords: list
    sections: list
    references: list          # list of Paper in final citation order (numbered 1..n)
    invalid_citations_removed: int = 0


# ------------------------------------------------------------------ references
def _initials(name: str) -> str:
    parts = name.replace(".", ". ").split()
    if len(parts) <= 1:
        return name.strip()
    first = " ".join(p[0].upper() + "." for p in parts[:-1] if p and p[0].isalpha())
    return f"{first} {parts[-1]}".strip()


def format_authors(authors: list) -> str:
    names = [_initials(a) for a in authors if a]
    if not names:
        return "Anonymous"
    if len(names) >= 4:
        return f"{names[0]} et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def ref_parts(p: Paper) -> dict:
    month = MONTHS[p.published.month - 1]
    idline = f"arXiv:{p.arxiv_id}" if p.arxiv_id else ""
    return {
        "authors": format_authors(p.authors),
        "title": p.title.rstrip("."),
        "venue": p.venue or "",
        "date": f"{month} {p.published.year}",
        "arxiv": idline,
        "doi": p.doi,
        "url": p.best_url(),
    }


def reference_markdown(p: Paper, n: int) -> str:
    r = ref_parts(p)
    out = f"[{n}] {r['authors']}, \u201c{r['title']},\u201d"
    if r["venue"]:
        out += f" *{r['venue']}*,"
    out += f" {r['date']}."
    if r["arxiv"]:
        out += f" {r['arxiv']}."
    if r["doi"]:
        out += f" doi: {r['doi']}."
    if r["url"]:
        out += f" [Online]. Available: {r['url']}"
    return out


def _bib_escape(text: str) -> str:
    return re.sub(r"([&%$#_])", r"\\\1", text or "")


def bibtex(references: list) -> str:
    keys, entries = Counter(), []
    for n, p in enumerate(references, start=1):
        last = re.sub(r"\W", "", (p.authors[0].split()[-1] if p.authors else "anon").lower()) or "anon"
        word = next((w for w in re.findall(r"[A-Za-z]+", p.title.lower()) if len(w) > 3), "paper")
        base = f"{last}{p.year}{word}"
        keys[base] += 1
        key = base if keys[base] == 1 else f"{base}{chr(96 + keys[base])}"
        authors = " and ".join(
            f"{a.split()[-1]}, {' '.join(a.split()[:-1])}".strip(", ") for a in p.authors if a.split()
        ) or "Anonymous"
        is_arxiv = p.is_preprint_only
        lines = [f"@{'misc' if is_arxiv else 'article'}{{{key},",
                 f"  author = {{{_bib_escape(authors)}}},",
                 f"  title = {{{{{_bib_escape(p.title)}}}}},"]
        if not is_arxiv:
            lines.append(f"  journal = {{{_bib_escape(p.venue)}}},")
        elif p.arxiv_id:
            lines.append(f"  howpublished = {{arXiv:{p.arxiv_id}}},")
        lines += [f"  year = {{{p.year}}},", f"  month = {{{p.published.strftime('%b').lower()}}},"]
        if p.doi:
            lines.append(f"  doi = {{{p.doi}}},")
        lines += [f"  url = {{{p.best_url()}}}", "}"]
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + "\n"


# ------------------------------------------------------------------ deterministic sections
def _d(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def methodology_blocks(*, papers, stats, queries, start, end, model, n_target) -> list:
    src_names = list(stats["raw_by_source"]) or ["no database"]
    blocks = [
        {"type": "h", "text": "Search strategy"},
        {"type": "p", "text": (
            f"Literature was retrieved programmatically from {', '.join(src_names)} using the public "
            f"APIs of these services. Only works dated between {_d(start)} and {_d(end)} were admitted; "
            "any earlier publication was excluded by a date filter applied both in the query and again "
            "in code after retrieval. The following queries were issued, each generated by a planning agent "
            "from the research topic:")},
    ] + [{"type": "li", "text": q} for q in queries] + [
        {"type": "h", "text": "Screening and inclusion"},
        {"type": "p", "text": (
            f"The searches returned {stats['raw_total']} records. {stats['dropped_date_or_abstract']} were removed "
            f"because they fell outside the date window or lacked a usable abstract, and {stats['duplicates']} duplicates "
            f"(matched by DOI, arXiv identifier or normalised title) were merged, leaving {stats['unique']} unique records. "
            f"A keyword screen on titles and abstracts removed {stats['irrelevant']} records that matched none of the "
            f"topic keywords. The remaining {stats['screened']} records were ranked by a relevance score (title matches "
            f"weighted twice as heavily as abstract matches, plus a small logarithmic citation term) and the top "
            f"{stats['included']} were included. Fig. 1 summarises this flow.")},
        {"type": "fig", "key": "flow",
         "caption": "Study identification and screening flow. All counts are produced by the retrieval pipeline of this run."},
        {"type": "h", "text": "Analysis"},
        {"type": "p", "text": (
            f"Titles and abstracts of the included studies were analysed by a pipeline of language-model agents "
            f"(a literature analyst that extracts contributions, methods and limitations, and an academic writer "
            f"that drafts the synthesis; model: {model}). Every citation in the text is checked against the retrieved "
            "records, and any citation that does not correspond to a retrieved study is deleted automatically.")},
        {"type": "h", "text": "Limitations of the method"},
        {"type": "p", "text": (
            "Only titles and abstracts were examined; full texts were not read, so methodological detail and nuance may be "
            "missing. Coverage is limited to the databases queried and to records that expose an abstract. Relevance "
            "screening is keyword based and may miss papers that use different terminology. The text was drafted by "
            "a language model and must be verified against the cited sources before any scholarly use.")},
    ]
    return blocks


def overview_blocks(*, papers, themes, order) -> list:
    n = len(papers)
    first, last = min(p.published for p in papers), max(p.published for p in papers)
    by_year = Counter(p.year for p in papers)
    year_txt = ", ".join(f"{y} ({c})" for y, c in sorted(by_year.items()))
    preprints = sum(p.is_preprint_only for p in papers)
    venues = Counter(p.venue for p in papers if not p.is_preprint_only)
    venue_txt = ""
    if venues:
        top = ", ".join(f"{v} ({c})" for v, c in venues.most_common(3))
        venue_txt = (f" {n - preprints} studies list a venue other than arXiv; the most frequent were {top}.")
    rows = []
    for p in sorted(papers, key=lambda p: order[p.id]):
        rows.append([f"[{order[p.id]}]", p.published.strftime("%Y-%m"), p.note])
    return [
        {"type": "p", "text": (
            f"The review covers {n} studies published between {first:%B %Y} and {last:%B %Y}. "
            f"By year of publication: {year_txt}. {preprints} of the {n} are available only as preprints "
            f"according to the retrieved metadata.{venue_txt} Fig. 2 shows the distribution of the included studies "
            "by publication quarter.")},
        {"type": "fig", "key": "timeline", "caption": "Included studies by publication quarter."},
        {"type": "p", "text": (
            f"The studies were grouped into {len(themes)} thematic clusters by the analyst agent (Fig. 3). "
            "A study may contribute to more than one cluster. Table I lists each study with a one-line summary of its "
            "main contribution as extracted from its abstract.")},
        {"type": "fig", "key": "themes",
         "caption": "Number of included studies per thematic cluster (studies may appear in several clusters)."},
        {"type": "table", "caption": "SUMMARY OF INCLUDED STUDIES", "header": ["Ref.", "Date", "Main contribution"],
         "rows": rows},
    ]


# ------------------------------------------------------------------ assembly
def assemble_paper(*, topic, plan: Plan, papers: list, analysis: Analysis, written: dict, stats: dict,
                   queries: list, start: date, end: date, model: str, author_line: str, today: date):
    """Return the finished ``PaperDraft`` with verified, renumbered citations."""
    mapper = CitationMapper(p.id for p in papers)

    def text_blocks(raw: str) -> list:
        out = []
        for kind, text in split_blocks(mapper.rewrite(raw)):
            out.append({"type": kind, "text": text})
        return out

    # citations are numbered in order of first appearance in the *written* sections
    intro = text_blocks(written.get("introduction", ""))
    synthesis = text_blocks(written.get("synthesis", ""))
    discussion = text_blocks(written.get("discussion", ""))
    future = text_blocks(written.get("future", ""))
    conclusion = text_blocks(written.get("conclusion", ""))
    for p in papers:           # studies never cited in prose still appear in Table I
        mapper.assign(p.id)

    for p in papers:
        c, m, _l = analysis.notes[p.id]
        p.note = c if c.endswith((".", "...")) else c + "."
    by_id = {p.id: p for p in papers}
    references = [by_id[i] for i, _n in sorted(mapper.order.items(), key=lambda kv: kv[1])]

    sections = [
        Section("Introduction", intro),
        Section("Methodology", methodology_blocks(papers=papers, stats=stats, queries=queries, start=start,
                                                  end=end, model=model, n_target=len(papers))),
        Section("Overview of Included Studies",
                overview_blocks(papers=papers, themes=analysis.themes, order=mapper.order)),
        Section("Thematic Synthesis", synthesis),
        Section("Discussion: Challenges and Limitations", discussion),
        Section("Research Gaps and Future Directions", future),
        Section("Conclusion", conclusion),
    ]
    sections = [s for s in sections if s.blocks]

    abstract = CitationMapper.strip_all(written.get("abstract", "")).replace("\n", " ").strip()
    draft = PaperDraft(
        title=plan.title,
        author_line=author_line or "AI Research Agent (automated literature synthesis)",
        byline=f"Generated {today:%d %B %Y} \u00b7 sources dated {start:%B %Y} onward \u00b7 model: {model}",
        abstract=abstract,
        keywords=plan.keywords[:6],
        sections=sections,
        references=references,
        invalid_citations_removed=mapper.removed,
    )
    return draft


def to_markdown(draft: PaperDraft, figure_files: dict) -> str:
    """Markdown version; figures are referenced as ``figures/<key>.png``."""
    out = [f"# {draft.title}", "", f"**{draft.author_line}**  ", f"*{draft.byline}*", "",
           f"**Abstract\u2014** {draft.abstract}", "", f"**Keywords\u2014** {', '.join(draft.keywords)}", ""]
    fig_no = table_no = 0
    letter = 0
    for i, sec in enumerate(draft.sections):
        out += [f"## {ROMAN[i]}. {sec.title}", ""]
        letter = 0
        for b in sec.blocks:
            t = b["type"]
            if t == "p":
                out += [b["text"], ""]
            elif t == "h":
                out += [f"### {chr(65 + letter)}. {b['text']}", ""]
                letter += 1
            elif t == "li":
                out += [f"- {b['text']}"]
            elif t == "fig":
                fig_no += 1
                out += ["", f"![Fig. {fig_no}]({figure_files.get(b['key'], 'figures/' + b['key'] + '.png')})", "",
                        f"*Fig. {fig_no}. {b['caption']}*", ""]
            elif t == "table":
                table_no += 1
                out += ["", f"**TABLE {ROMAN[table_no - 1]}: {b['caption']}**", "",
                        "| " + " | ".join(b["header"]) + " |", "|" + "---|" * len(b["header"])]
                out += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in b["rows"]] + [""]
        if out and out[-1] != "":
            out.append("")
    out += ["## References", ""]
    out += [reference_markdown(p, n) + "  " for n, p in enumerate(draft.references, start=1)]
    out += ["", "---", "*This paper was generated by an AI agent from titles and abstracts of retrieved publications. "
            "Verify every claim and reference against the original sources before use.*", ""]
    return "\n".join(out)
