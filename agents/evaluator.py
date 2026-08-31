import logging
import json
import re

from utils.llm import get_llm

logger = logging.getLogger(__name__)
llm = get_llm(temperature=0)


def evaluator_agent(topic, research):
    if topic is None:
        raise ValueError("Evaluator topic cannot be None.")

    if research is None:
        raise ValueError("Evaluator research cannot be None.")

    topic = str(topic).strip()
    research = str(research).strip()

    if not topic:
        raise ValueError("Evaluator topic cannot be empty.")

    if not research:
        raise ValueError("Evaluator research cannot be empty.")

    logger.info("Evaluator started | topic=%s", topic)
    logger.info("Evaluating research | research_length=%d", len(research))

    prompt = f"""
You are evaluating research quality.

Topic:
{topic}

Research:
{research}

Rate the research from 0 to 10.

Consider:
- Does it answer the topic?
- Is important information missing?
- Is the information specific and useful?
- Are the sources/reasoning adequate?

Return:
{{"score": <integer from 0 to 10>, "feedback": "<explanation>"}}

Return only valid JSON.
""".strip()

    try:
        response = llm.invoke(prompt)

    except Exception as error:
        logger.exception("Evaluator LLM call failed: %s", error)
        raise RuntimeError("Evaluator failed to evaluate research.") from error

    if not response or not response.content:
        logger.error("Evaluator returned an empty response.")
        raise RuntimeError("Evaluator returned an empty response.")

    result = response.content.strip()

    # Prefer the requested JSON format, but tolerate the labelled format that
    # smaller local models sometimes return despite the instruction.
    try:
        evaluation = json.loads(result.removeprefix("```json").removesuffix("```").strip())
        score = int(evaluation["score"])
        feedback = str(evaluation["feedback"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        score_match = re.search(r"^\s*score\s*:\s*(\d+)", result, re.IGNORECASE | re.MULTILINE)
        feedback_match = re.search(
            r"^\s*feedback\s*:\s*(.+)",
            result,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if not score_match or not feedback_match:
            logger.error("Evaluator returned an invalid format: %r", result)
            raise RuntimeError("Evaluator returned an invalid evaluation format.")

        score = int(score_match.group(1))
        feedback = feedback_match.group(1).strip()

    if not 0 <= score <= 10:
        raise RuntimeError("Evaluator returned a score outside the range 0 to 10.")

    if not feedback:
        raise RuntimeError("Evaluator returned empty feedback.")

    logger.info("Evaluator completed | output_length=%d", len(result))

    return {"score": score, "feedback": feedback}
