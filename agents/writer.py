import logging
from utils.llm import get_llm

logger = logging.getLogger(__name__)
llm = get_llm(temperature=0.2)

def writer_agent(topic, summary):
    if topic is None:
        raise ValueError(
            "Writer topic cannot be None."
        )

    topic = str(topic).strip()
    if not topic:
        raise ValueError(
            "Writer topic cannot be empty."
        )

    if summary is None:
        raise ValueError(
            "Writer summary cannot be None."
        )

    summary = str(summary).strip()

    if not summary:
        raise ValueError(
            "Writer summary cannot be empty."
        )

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

Use only information supported by the summary.
Do not invent statistics, sources, or facts.
""".strip()

    logger.info(
        "Writer started | topic_length=%d | summary_length=%d",
        len(topic),
        len(summary)
    )

    response = llm.invoke(prompt)

    if not response or not response.content:
        raise RuntimeError(
            "Writer returned an empty response."
        )

    result = response.content.strip()

    if not result:
        raise RuntimeError(
            "Writer returned an empty response."
        )

    logger.info(
        "Writer completed | output_length=%d",
        len(result)
    )

    return result