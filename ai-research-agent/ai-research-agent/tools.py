from ddgs import DDGS
from crewai.tools import tool


@tool("DuckDuckGo Web Search")
def search_web(query: str) -> str:
    """Search DuckDuckGo and return result titles, URLs, and snippets."""
    try:
        results = DDGS().text(
            query,
            max_results=8,
            backend="duckduckgo",
        )

        if not results:
            return "No search results were found. Try a different query."

        formatted = []
        for index, result in enumerate(results, start=1):
            formatted.append(
                f"Source {index}\n"
                f"Title: {result.get('title', 'No title')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Snippet: {result.get('body', 'No description')}"
            )
        return "\n\n".join(formatted)

    except Exception as exc:
        return f"DuckDuckGo search failed: {type(exc).__name__}: {exc}"
