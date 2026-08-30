import logging
import os
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()
logger = logging.getLogger(__name__)
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10
MAX_QUERY_LENGTH = 1000


def _get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_key":

        raise RuntimeError(
            "TAVILY_API_KEY is missing. "
            "Add your Tavily API key to .env."
        )
    return TavilyClient(
        api_key=api_key
    )

def web_search(query,max_results=DEFAULT_MAX_RESULTS):
    
    if query is None:

        raise ValueError(
            "Search query cannot be None."
        )

    query = str(query).strip()

    if not query:

        raise ValueError(
            "Search query cannot be empty."
        )

    if len(query) > MAX_QUERY_LENGTH:

        raise ValueError(
            f"Search query cannot exceed "
            f"{MAX_QUERY_LENGTH} characters."
        )

    if not isinstance(
        max_results,
        int
    ):

        raise TypeError(
            "max_results must be an integer."
        )

    if not 1 <= max_results <= MAX_ALLOWED_RESULTS:

        raise ValueError(
            f"max_results must be between "
            f"1 and {MAX_ALLOWED_RESULTS}."
        )

    logger.info(
        "Web search started | query=%s | max_results=%d",
        query,
        max_results
    )

    try:

        client = _get_tavily_client()

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )

    except Exception as error:

        logger.exception(
            "Tavily search failed: %s",
            error
        )

        raise RuntimeError(
            f"Tavily search failed for query: "
            f"{query}"
        ) from error

    if not isinstance(
        response,
        dict
    ):

        raise RuntimeError(
            "Tavily returned an invalid response."
        )

    raw_results = response.get(
        "results",
        []
    )

    if not isinstance(
        raw_results,
        list
    ):

        raise RuntimeError(
            "Tavily results field is invalid."
        )

    if not raw_results:

        logger.warning(
            "No search results found | query=%s",
            query
        )

        return []

    results = []

    for item in raw_results:

        if not isinstance(
            item,
            dict
        ):

            continue

        title = str(
            item.get(
                "title",
                "Untitled"
            )
        ).strip()

        url = str(
            item.get(
                "url",
                ""
            )
        ).strip()

        content = str(
            item.get(
                "content",
                ""
            )
        ).strip()

        score = item.get(
            "score"
        )

        # Skip completely empty records
        if not title and not url and not content:

            continue

        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
            }
        )
    if not results:

        logger.warning(
            "Tavily returned no usable results | query=%s",
            query
        )

        return []

    logger.info(
        "Web search completed | query=%s | results=%d",
        query,
        len(results)
    )

    return results

def format_search_results(results):
    if not results:

        return "No search results found."

    formatted = []

    for index, result in enumerate(
        results,
        start=1
    ):

        formatted.append(
            f"""
SOURCE {index}

Title:
{result.get("title", "Untitled")}

URL:
{result.get("url", "")}

Content:
{result.get("content", "")}
""".strip()
        )

    return "\n\n".join(
        formatted
    )