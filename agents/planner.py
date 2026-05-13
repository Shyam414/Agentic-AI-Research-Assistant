from agents.llm import get_llm

llm = get_llm(temperature=0)


def planner_agent(topic, memory=""):
    prompt = f"""
You are a planning agent for a multi-agent research assistant.

Create a focused research plan for this topic:
{topic}

Previous session context:
{memory or "No previous context."}

Return:
1. Main areas to research
2. Important questions to answer
3. Key technologies, people, organizations, or trends to check
4. Search queries the research agent should use
"""

    response = llm.invoke(prompt)
    return response.content
