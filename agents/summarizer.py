import logging
from utils.llm import get_llm

logger = logging.getLogger(__name__)
llm = get_llm(temperature=0)

def summarizer_agent(research_data):

    if research_data is None:
        raise ValueError(
            "Research data cannot be None."
        )

    research_data = str(
        research_data
    ).strip()

    if not research_data:
        raise ValueError(
            "Research data cannot be empty."
        )

    prompt = f"""
Summarize the following research clearly.

Research:

{research_data}

Give concise bullet points with:

- Core findings
- Supporting evidence
- Notable sources
- Gaps or uncertainties

Only use information contained in the research.
Do not invent facts.
""".strip()

    logger.info(
        "Summarizer started | input_length=%d",
        len(research_data)
    )
    response = llm.invoke(prompt)

    if not response or not response.content:
        raise RuntimeError(
            "Summarizer returned an empty response."
        )

    result = response.content.strip()

    if not result:
        raise RuntimeError(
            "Summarizer returned an empty response."
        )

    logger.info(
        "Summarizer completed | output_length=%d",
        len(result)
    )

    return result