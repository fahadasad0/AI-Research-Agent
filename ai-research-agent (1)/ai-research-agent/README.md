# AI Research Paper Agent

A multi-agent literature-review generator (CrewAI + Groq + Streamlit) that writes an
IEEE-style research paper **only from publications dated 2024 or later**, with a verified
reference list and figures drawn from the retrieved data.

## What it produces

* Two-column A4 **PDF** with title, abstract, keywords, numbered sections, three figures,
  a summary table and IEEE-style references
* **Markdown**, **BibTeX** and a **ZIP bundle** (PDF + Markdown + BibTeX + PNG figures)
* In-app preview with the source list, search queries and screening counts

## How it works

| Step | Who | What |
|---|---|---|
| 1 | Research Strategist (LLM agent) | topic -> title, keywords, 4 search queries |
| 2 | Retrieval (plain code) | arXiv, Semantic Scholar and (with a key) OpenAlex; **hard date gate >= 2024**, dedupe, keyword screening, ranking |
| 3 | Literature Analyst (LLM agent) | per-paper contribution / method / limitation + 3-5 themes |
| 4 | Academic Writer (LLM agent) | Introduction, Synthesis, Discussion, Future work, Abstract, Conclusion |
| 5 | Verifier (plain code) | deletes any `[n]` that is not a retrieved paper, renumbers citations in order of appearance |
| 6 | Publisher (plain code) | figures + PDF + Markdown + BibTeX |

Retrieval is deliberately **not** an LLM tool call. References come from structured API
metadata (authors, date, DOI, URL), so the model cannot invent or mangle a citation.

Figures (all computed from the run, nothing invented): screening flow diagram (PRISMA style),
papers per publication quarter, papers per thematic cluster.

## Fixing `'ascii' codec can't encode characters in position 63-94`

HTTP headers must be ASCII. The header is `Authorization: Bearer <key>`: 7 characters plus a
56-character Groq key = 63. "Position 63" therefore means there are non-ASCII characters
*after* the key: curly quotes, zero-width spaces, or other text pasted into the secret.

* `clean_api_key()` strips quotes, spaces, `Bearer `, a pasted `GROQ_API_KEY=` prefix and invisible
  characters, and if anything non-ASCII remains it tells you exactly which characters (Unicode name
  and position) instead of crashing inside litellm.
* The `RuntimeError: no running event loop` above it in your traceback is **not** the cause. It is
  CrewAI's normal fallback to `asyncio.run`.

Streamlit Cloud > App settings > Secrets, plain straight quotes and nothing after the key:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENALEX_API_KEY = ""   # optional
```

## Groq free-tier limits (important)

`openai/gpt-oss-120b` on the free plan is limited to roughly 8,000 tokens per minute and 200,000
per day (check https://console.groq.com/settings/limits; limits are per organisation).
The app therefore:

* paces requests with a sliding-window token budget (sidebar: "Tokens-per-minute budget"),
* keeps prompts compact (abstract excerpts, then short analyst notes, never full abstracts twice),
* uses 6 LLM calls per paper (about 15-20K tokens), so expect ~3-6 minutes and roughly 8-10 papers per day on the free plan,
* stops with a clear message when the daily quota is exhausted.

On the paid Developer tier raise the budget in the sidebar (for example 200000).

## Data sources

* **arXiv** - no key, always used.
* **Semantic Scholar** - no key required (shared rate limit; a failure only produces a warning).
* **OpenAlex** - since Feb 2026 it requires a free API key (https://openalex.org/settings/api).
  Add `OPENALEX_API_KEY` to enable it.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env     # then edit .env
streamlit run app.py
```

Offline tests (no network, no API key): `pip install pytest && python -m pytest -q`

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub (check `git status`: `.env` must not be staged).
2. share.streamlit.io > New app > main file `app.py`, Python 3.12.
3. Add the secrets shown above, deploy.

## Honest limitations

* Only **titles and abstracts** are read. Claims about methods or results are as deep as an abstract.
* The text is AI-written. Verify claims against the original papers before any scholarly use, and
  follow your institution's rules on disclosing AI assistance.
* Keyword screening can miss papers that use different terminology.
* The "2024 or later" rule uses each record's publication / first-submission date from the APIs.
  A journal version of a preprint first posted in 2023 can appear under its 2024 journal date.

## Ideas for next steps

* Read open-access full texts (arXiv PDFs) for deeper analysis
* Pull figures from CC-BY papers with proper attribution (only after checking each licence)
* A reviewer agent that re-checks each claim against its cited abstract

## License

MIT
