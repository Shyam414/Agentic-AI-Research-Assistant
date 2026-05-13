import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def web_search(query, max_results=5):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_key":
        return "Tavily API key is missing. Add TAVILY_API_KEY to your .env file."

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
    )

    results = []
    answer = response.get("answer")
    if answer:
        results.append(f"Answer: {answer}\n")

    for result in response.get("results", []):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        results.append(f"Title: {title}\nURL: {url}\nContent: {content}\n")

    return "\n".join(results) if results else "No search results found."
