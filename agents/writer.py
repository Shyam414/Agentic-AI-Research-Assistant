from agents.llm import get_llm

llm = get_llm(temperature=0.2)


def writer_agent(topic, summary):
    prompt = f"""
Write a professional research report.

Topic:
{topic}

Summary:
{summary}

Include:
- Introduction
- Key Insights
- Practical Applications
- Future Scope
- Conclusion
"""

    response = llm.invoke(prompt)
    return response.content
