from agents.llm import get_llm

llm = get_llm(temperature=0)


def summarizer_agent(research_data):
    prompt = f"""
Summarize the following research clearly.

Research:
{research_data}

Give concise bullet points with:
- Core findings
- Supporting evidence
- Notable sources
- Gaps or uncertainties
"""

    response = llm.invoke(prompt)
    return response.content
