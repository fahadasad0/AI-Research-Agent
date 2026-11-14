import os
import streamlit as st
from dotenv import load_dotenv
from research_agent import generate_research_report

load_dotenv()

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Agent")
st.caption("A single-agent research assistant powered by CrewAI, Groq, and DuckDuckGo.")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "This app uses Groq's `openai/gpt-oss-120b` model. "
        "Your API key is read from Streamlit Secrets or a local `.env` file."
    )

try:
    secret_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_key = ""

api_key = secret_key or os.getenv("GROQ_API_KEY", "")

topic = st.text_area(
    "Research topic",
    placeholder="e.g. Applications of agentic AI in scientific research",
    height=110,
)

col1, col2 = st.columns([1, 3])
with col1:
    generate = st.button("Generate report", type="primary", use_container_width=True)
with col2:
    st.caption("Research may take several minutes depending on search results and model response time.")

if generate:
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to `.env` locally or Streamlit Cloud → App settings → Secrets.")
    elif not topic.strip():
        st.warning("Enter a research topic first.")
    else:
        try:
            with st.spinner("The research agent is searching, analyzing, and writing..."):
                report = generate_research_report(topic.strip(), api_key)
            st.success("Report generated.")
            st.markdown(report)
            st.download_button(
                "Download Markdown report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as exc:
            st.error("The report could not be generated.")
            st.exception(exc)

st.divider()
st.caption("Research outputs are AI-generated. Verify important claims and references against the original sources.")
