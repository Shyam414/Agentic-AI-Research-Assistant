import logging
import os


from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS=5
MAX_ALLOWED_RESULTS=10
MAX_QUERY_LENGTH = 1000


def _get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key or api_key == "your_tavily_key":
        raise RuntimeError("TAVILY_API_KEY is missing. Add your Tavily API key to .env.")

    return TavilyClient(api_key=api_key)


def _validate_search_query(query):
    if query is None:
        raise ValueError("Search query cannot be None.")

    query = str(query).strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Search query cannot exceed {MAX_QUERY_LENGTH} characters.")

    return query


def _validate_max_results(max_results):
    if not isinstance(max_results, int):
        raise TypeError("max_results must be an integer.")

    if not 1 <= max_results <= MAX_ALLOWED_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_ALLOWED_RESULTS}.")

    return max_results


def _extract_search_results(response):
    if not isinstance(response, dict):
        raise RuntimeError("Tavily returned an invalid response.")

    results = response.get("results", [])

    if not isinstance(results, list):
        raise RuntimeError("Tavily results field is invalid.")

    return results


def _normalize_search_result(item):
    if not isinstance(item, dict):
        return None

    title = str(item.get("title", "Untitled")).strip()
    url = str(item.get("url", "")).strip()
    content = str(item.get("content", "")).strip()
    score = item.get("score")

    if not title and not url and not content:
        return None

    return {"title": title, "url": url, "content": content, "score": score}


def web_search(query, max_results=DEFAULT_MAX_RESULTS):
    query = _validate_search_query(query)
    max_results = _validate_max_results(max_results)

    logger.info("Web search started | query=%s | max_results=%d", query, max_results)

    try:
        client = _get_tavily_client()
        response = client.search(query=query, search_depth="advanced", max_results=max_results, include_answer=True)
        raw_results = _extract_search_results(response)
    except Exception as error:
        logger.exception("Tavily search failed: %s", error)
        raise RuntimeError(f"Tavily search failed for query: {query}") from error

    if not raw_results:
        logger.warning("No search results found | query=%s", query)
        return []

    results = [result for item in raw_results if (result := _normalize_search_result(item)) is not None]

    if not results:
        logger.warning("Tavily returned no usable results | query=%s", query)
        return []

    logger.info("Web search completed | query=%s | results=%d", query, len(results))
    return results


# def format_search_results(results):
#     if not results:
#         return "No search results found."

#     formatted = []

#     for index, result in enumerate(results, start=1):
#         formatted.append(
#             f"SOURCE {index}\n\n"
#             f"Title:\n{result.get('title', 'Untitled')}\n\n"
#             f"URL:\n{result.get('url', '')}\n\n"
#             f"Content:\n{result.get('content', '')}"
#         )

#     return "\n\n".join(formatted)