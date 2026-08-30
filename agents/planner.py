import json
import logging
from utils.llm import get_llm

logger = logging.getLogger(__name__)
llm = get_llm(temperature=0)


def _parse_plan(raw_output, topic):
    candidate = raw_output.strip()

    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3].rstrip()

    try:
        plan = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Planner returned invalid JSON.")
        try:
            plan = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise RuntimeError("Planner returned invalid JSON.") from error

    if not isinstance(plan, dict):
        raise RuntimeError("Planner response must be a JSON object.")

    nested_plan = plan.get("research_plan")
    if isinstance(nested_plan, dict):
        plan = nested_plan

    normalized = {}
    for key in ("research_areas", "questions", "technologies", "search_queries"):
        values = plan.get(key, [])
        if not isinstance(values, list):
            raise RuntimeError(f"Planner field '{key}' must be a list.")
        normalized[key] = [str(value).strip() for value in values if str(value).strip()]

    # A usable fallback lets research continue when a small local model omits this field.
    if not normalized["search_queries"]:
        logger.warning("Planner omitted search queries; using topic-based fallback queries.")
        normalized["search_queries"] = [
            topic,
            f"{topic} latest developments",
            f"{topic} benefits risks and challenges",
        ]

    return normalized

def planner_agent(topic, memory=""):

    if topic is None:
        raise ValueError("Planner topic cannot be None.")

    topic = str(topic).strip()

    if not topic:
        raise ValueError("Planner topic cannot be empty.")

    memory = str(memory or "").strip()

    if len(topic) > 5000:
        raise ValueError(
            "Planner topic is too long."
        )

    if len(memory) > 10000:
        raise ValueError(
            "Planner memory is too long."
        )

    prompt = f"""
You are a planning agent for a multi-agent research assistant.
Create a focused research plan for this topic:
{topic}
Previous session context:
{memory or "No previous context."}
Return ONLY valid JSON in this exact structure:
{{
    "research_areas": ["area"],
    "questions": ["question"],
    "technologies": ["technology"],
    "search_queries": ["specific web search query"]
}}
Requirements:
- research_areas: important areas to investigate
- questions: important questions the research should answer
- technologies: relevant technologies, people, organizations, or trends
- search_queries: specific web search queries for the research agent
Do not include markdown.
Do not include explanations outside the JSON.
""".strip()
    logger.info(
        "Planner started | topic_length=%d",
        len(topic)
    )
    response = llm.invoke(prompt, json_mode=True)

    if not response or not response.content:
        raise RuntimeError(
            "Planner returned an empty response."
        )

    raw_output = response.content.strip()

    try:
        plan = _parse_plan(raw_output, topic)
    except RuntimeError:
        logger.error("Planner returned invalid JSON: %s", raw_output[:500])
        raise

    logger.info(
        "Planner completed | search_queries=%d",
        len(plan["search_queries"])
    )

    return plan
