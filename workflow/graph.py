from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.writer import writer_agent
from agents.evaluator import evaluator_agent

class AgentState(TypedDict, total=False):
    topic: str
    memory: str
    plan: dict
    research: str
    quality_score: int
    evaluation_feedback: str
    summary: str
    report: str


def planner_node(state: AgentState):
    plan = planner_agent(state["topic"], state.get("memory", ""))
    return { "plan": plan}

def researcher_node(state: AgentState):
    research = researcher_agent(state["topic"],state["plan"])
    return {"research": research}

def evaluator_node(state: AgentState):
    evaluation = evaluator_agent(
        state["topic"],
        state["research"]
    )

    return {
        "quality_score": evaluation["score"],
        "evaluation_feedback": evaluation["feedback"]
    }

def route_after_evaluation(state: AgentState):

    if state["quality_score"] < 7:
        return "researcher"

    return "summarizer"

def summarizer_node(state: AgentState):
    summary = summarizer_agent(state["research"])
    return { "summary": summary}

def writer_node(state: AgentState):
    report = writer_agent(state["topic"],state["summary"])
    return {"report": report}

workflow = StateGraph(AgentState)

workflow.add_node("planner",planner_node)
workflow.add_node("researcher",researcher_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("summarizer",summarizer_node)
workflow.add_node("writer",writer_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner","researcher")
workflow.add_edge("researcher", "evaluator")
workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "researcher": "researcher",
        "summarizer": "summarizer"
    }
)
workflow.add_edge("summarizer","writer")
workflow.add_edge("writer",END)
app = workflow.compile(checkpointer=MemorySaver())