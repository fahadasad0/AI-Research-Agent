import os
from crewai import Agent, Crew, LLM, Process, Task
from tools import search_web


MODEL_NAME = "groq/openai/gpt-oss-120b"


def create_research_crew(topic: str, api_key: str) -> Crew:
    """Create a single-agent CrewAI research workflow."""
    os.environ["GROQ_API_KEY"] = api_key

    llm = LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        max_tokens=8192,
    )

    researcher = Agent(
        role="AI Research Analyst",
        goal=(
            "Research the user's topic using web search, assess the "
            "available evidence, and write a useful, well-structured report."
        ),
        backstory=(
            "You are a careful research analyst. You distinguish evidence "
            "from speculation, prefer primary and reputable sources, and "
            "never invent papers, authors, dates, statistics, or citations."
        ),
        llm=llm,
        tools=[search_web],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )

    task = Task(
        description=f"""
Research this topic: {topic}

Workflow:
1. Break the topic into several focused search questions.
2. Use the DuckDuckGo search tool for relevant queries.
3. Prefer peer-reviewed papers, official documentation, government or
   institutional sources, and reputable technical publications.
4. Use multiple independent sources where possible.
5. Clearly distinguish what a source says from your own synthesis.
6. Do not invent details or cite sources that were not returned by search.
7. Search-result snippets are leads, not proof that the full page was read.
   State this limitation where relevant.
8. Identify disagreements, limitations, and evidence gaps.

Write a Markdown report with:
# Research title
## Abstract
## 1. Introduction
## 2. Background and key concepts
## 3. Literature and existing approaches
## 4. Findings and analysis
## 5. Challenges and limitations
## 6. Research gaps
## 7. Future research directions
## 8. Conclusion
## References

For every reference, include the exact URL returned by the search tool.
Do not fabricate authors, publication dates, paper titles, DOIs, or URLs.
If evidence is insufficient, say so explicitly.
""",
        expected_output=(
            "A structured Markdown research report with a clear abstract, "
            "critical analysis, limitations, research gaps, conclusion, "
            "and a list of source URLs actually returned by web search."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def generate_research_report(topic: str, api_key: str) -> str:
    crew = create_research_crew(topic, api_key)
    result = crew.kickoff()
    return str(result.raw)
