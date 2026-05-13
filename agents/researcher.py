from agents.llm import get_llm
from tools.search_tool import web_search

llm = get_llm(temperature=0)


def researcher_agent(topic, plan):
    search_results = web_search(topic)

    prompt = f"""
You are a research agent.

Topic:
{topic}

Research plan:
{plan}

Web results:
{search_results}

Extract the most important information. Keep source URLs when available.
Separate facts, trends, opportunities, risks, and open questions.
"""

    response = llm.invoke(prompt)
    return response.content
