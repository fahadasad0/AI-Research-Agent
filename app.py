import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_client import ApiKeyError, DailyQuotaError, clean_api_key
from paper_builder import ROMAN
from research_agent import NotEnoughPapers, generate_paper
from sources import MIN_YEAR

load_dotenv()

st.set_page_config(page_title="AI Research Paper Agent", page_icon="🔬", layout="wide")
st.title("🔬 AI Research Paper Agent")
st.caption(
    "Multi-agent literature review that writes a research paper from **real publications dated "
    f"{MIN_YEAR} or later**, with verified references and figures drawn from the retrieved data."
)


def secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets.toml locally
        value = ""
    return value or os.getenv(name, "")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    typed_key = st.text_input("Groq API key (optional)", type="password",
                              help="Leave empty to use GROQ_API_KEY from Streamlit Secrets or .env.")
    model = st.text_input("Groq model", value="openai/gpt-oss-120b")
    n_papers = st.slider("Papers to include", 6, 20, 12,
                         help="More papers = longer prompts. On Groq's free tier keep this at 12 or below.")
    min_year = st.number_input("Earliest publication year", min_value=MIN_YEAR, max_value=date.today().year,
                               value=MIN_YEAR, step=1,
                               help=f"Papers published before {MIN_YEAR} are never used.")
    tpm = st.number_input("Tokens-per-minute budget", min_value=2000, max_value=500_000, value=7000, step=1000,
                          help="Groq free tier: ~8,000 for gpt-oss-120b. Raise it (e.g. 200000) on the paid Developer tier.")
    author_line = st.text_input("Author line (optional)", placeholder="e.g. Your Name, Your University")
    st.divider()
    st.markdown("**Optional data sources**")
    st.caption("arXiv and Semantic Scholar need no key. OpenAlex now requires a free key.")
    openalex_key = st.text_input("OpenAlex API key", type="password", value=secret("OPENALEX_API_KEY"))
    use_s2 = st.checkbox("Also search Semantic Scholar", value=True)

topic = st.text_area("Research topic", height=100,
                     placeholder="e.g. Agentic AI systems for automating scientific research")
c1, c2 = st.columns([1, 3])
go = c1.button("Generate paper", type="primary", use_container_width=True)
c2.caption("Takes roughly 3 to 6 minutes on Groq's free tier because requests are paced to respect its token limit.")

# ------------------------------------------------------------------ run
if go:
    st.session_state.pop("result", None)
    if not topic.strip():
        st.warning("Enter a research topic first.")
    else:
        try:
            api_key = clean_api_key(typed_key or secret("GROQ_API_KEY"))
        except ApiKeyError as exc:
            st.error(str(exc))
            st.stop()

        bar = st.progress(0.0)
        state = {"frac": 0.0}
        with st.status("The research agents are working...", expanded=True) as status:
            def progress(message: str, fraction: float):
                if fraction >= 0:
                    state["frac"] = fraction
                bar.progress(min(state["frac"], 1.0))
                status.write(message)

            try:
                st.session_state["result"] = generate_paper(
                    topic, api_key=api_key, model=model.strip(), n_papers=int(n_papers), min_year=int(min_year),
                    tokens_per_minute=int(tpm), author_line=author_line.strip(),
                    openalex_key=openalex_key.strip(), s2_key=secret("S2_API_KEY"),
                    use_semantic_scholar=use_s2, progress=progress)
                status.update(label="Paper generated", state="complete", expanded=False)
            except ApiKeyError as exc:
                status.update(label="API key problem", state="error")
                st.error(str(exc))
            except DailyQuotaError as exc:
                status.update(label="Daily quota reached", state="error")
                st.error(str(exc))
            except NotEnoughPapers as exc:
                status.update(label="Not enough papers found", state="error")
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                status.update(label="Generation failed", state="error")
                st.error("The paper could not be generated.")
                st.exception(exc)

# ------------------------------------------------------------------ results (survive reruns)
res = st.session_state.get("result")
if res:
    st.success(f"Paper ready: {len(res.papers)} papers from {min(p.published for p in res.papers):%B %Y} "
               f"to {max(p.published for p in res.papers):%B %Y}, {len(res.draft.references)} references.")
    for w in res.warnings:
        st.warning(w)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ PDF (two-column paper)", res.pdf, "research_paper.pdf", "application/pdf",
                       use_container_width=True)
    d2.download_button("⬇ Bundle (PDF + Markdown + BibTeX + figures)", res.bundle, "research_paper_bundle.zip",
                       "application/zip", use_container_width=True)
    d3.download_button("⬇ BibTeX references", res.bibtex, "references.bib", "text/plain",
                       use_container_width=True)

    tab_paper, tab_sources, tab_method = st.tabs(["Paper preview", "Source papers", "Retrieval details"])

    with tab_paper:
        draft = res.draft
        st.header(draft.title)
        st.markdown(f"**{draft.author_line}**  \n*{draft.byline}*")
        st.markdown(f"**Abstract—** {draft.abstract}")
        st.markdown(f"**Keywords—** {', '.join(draft.keywords)}")
        fig_no = 0
        for i, sec in enumerate(draft.sections):
            st.subheader(f"{ROMAN[i]}. {sec.title}")
            letter = 0
            for b in sec.blocks:
                if b["type"] == "p":
                    st.markdown(b["text"])
                elif b["type"] == "h":
                    st.markdown(f"**{chr(65 + letter)}. {b['text']}**")
                    letter += 1
                elif b["type"] == "li":
                    st.markdown(f"- {b['text']}")
                elif b["type"] == "fig":
                    fig_no += 1
                    st.image(res.figures[b["key"]], caption=f"Fig. {fig_no}. {b['caption']}", width=520)
                elif b["type"] == "table":
                    st.caption(f"TABLE: {b['caption']}")
                    st.dataframe(pd.DataFrame(b["rows"], columns=b["header"]), hide_index=True,
                                 use_container_width=True)
        st.subheader("References")
        from paper_builder import reference_markdown
        for n, p in enumerate(draft.references, start=1):
            st.markdown(reference_markdown(p, n))

    with tab_sources:
        rows = [{"Ref": n, "Date": p.published.isoformat(), "Title": p.title,
                 "Authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                 "Venue": p.venue, "Found in": ", ".join(p.sources), "Link": p.best_url()}
                for n, p in enumerate(res.draft.references, start=1)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Link": st.column_config.LinkColumn("Link")})

    with tab_method:
        st.write("**Search queries**")
        for q in res.queries:
            st.markdown(f"- {q}")
        st.write("**Screening counts**")
        st.json(res.stats)
        st.write("**Analyst themes**")
        for name, ids, summary in res.analysis.themes:
            st.markdown(f"- **{name}** ({len(ids)} papers): {summary}")

st.divider()
st.caption("Generated text is a draft built from titles and abstracts. Verify every claim and reference "
           "against the original papers before using it in scholarly work.")
