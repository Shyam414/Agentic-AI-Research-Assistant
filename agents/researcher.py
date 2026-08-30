import logging
from utils.llm import get_llm
from utils.search_tool import web_search

logger = logging.getLogger(__name__)
llm = get_llm(temperature=0)

def researcher_agent(topic, plan):

    if topic is None:
        raise ValueError(
            "Researcher topic cannot be None."
        )

    topic = str(topic).strip()

    if not topic:
        raise ValueError(
            "Researcher topic cannot be empty."
        )

    if not isinstance(plan, dict):
        raise TypeError(
            "Research plan must be a dictionary."
        )

    search_queries = plan.get(
        "search_queries"
    )

    if not isinstance(
        search_queries,
        list
    ):

        raise ValueError(
            "Research plan must contain "
            "a search_queries list."
        )

    if not search_queries:

        raise ValueError(
            "Research plan contains no "
            "search queries."
        )

    logger.info(
        "Researcher started | queries=%d",
        len(search_queries)
    )

    all_results = []

    for query in search_queries:

        query = str(query).strip()

        if not query:
            continue

        logger.info(
            "Executing research query: %s",
            query
        )

        try:

            results = web_search(query)

            if results:
                all_results.append(
                    f"QUERY: {query}\n"
                    f"RESULTS:\n{results}"
                )

        except Exception as error:

            logger.warning(
                "Search failed for query '%s': %s",
                query,
                error
            )

    if not all_results:

        raise RuntimeError(
            "Researcher could not obtain "
            "any web search results."
        )

    search_results = "\n\n".join(
        all_results
    )

    prompt = f"""
You are a research agent.

Topic:
{topic}

Research plan:
{plan}

Web results:
{search_results}

Extract the most important information.

Keep source URLs when available.

Separate:

- Facts
- Trends
- Opportunities
- Risks
- Open questions

Do not invent information that is not supported
by the provided research results.
""".strip()

    response = llm.invoke(prompt)

    if not response or not response.content:
        raise RuntimeError(
            "Researcher returned an empty response."
        )

    result = response.content.strip()

    if not result:
        raise RuntimeError(
            "Researcher returned an empty response."
        )

    logger.info(
        "Researcher completed | output_length=%d",
        len(result)
    )

    return result