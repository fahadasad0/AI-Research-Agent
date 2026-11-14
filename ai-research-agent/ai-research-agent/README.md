# AI Research Agent

A beginner-friendly, single-agent research assistant built with CrewAI, Groq's
`openai/gpt-oss-120b` model, DuckDuckGo search (DDGS), and Streamlit.

## Features

- Accepts a research topic in a Streamlit interface.
- Uses one CrewAI agent (not a multi-agent crew).
- Searches DuckDuckGo for relevant web results.
- Produces a structured Markdown research report.
- Allows downloading the report.

## Requirements

- Python 3.12 recommended
- A Groq API key
- Internet access for DuckDuckGo and Groq

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-agent.git
cd ai-research-agent
```

## 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The direct dependencies are pinned for repeatability. Their transitive dependency
resolution should still be tested in a clean environment before production use.

## 4. Configure Groq

Copy `.env.example` to `.env` and add your API key:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Get a key from https://console.groq.com/keys

Never commit `.env` or publish your API key.

## 5. Run locally

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in at https://share.streamlit.io/.
3. Create an app and select your repository and `main` branch.
4. Set the app's main file to `app.py`.
5. Select Python 3.12 in the deployment settings if the option is available.
6. In the app's Settings → Secrets, add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

7. Deploy and inspect the application logs if startup fails.

## GitHub upload

```bash
git init
git add .
git commit -m "Initial AI Research Agent"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-research-agent.git
git push -u origin main
```

Before pushing, check that `.env` is not staged:

```bash
git status
```

## Important research limitations

- DuckDuckGo snippets and URLs are discovery results, not proof that the
  underlying pages or full papers were read.
- The agent can make mistakes or produce incomplete references. Verify every
  important claim against the original source.
- Search availability, API quotas, and model availability may change.
- This is an educational MVP, not a validated academic literature-review system.

## Suggested next improvements

1. Extract and read the full text of source webpages.
2. Add PDF paper ingestion and DOI metadata validation.
3. Add citation verification and source-quality scoring.
4. Evaluate factual accuracy, citation correctness, coverage, latency, and cost.

## License

MIT
