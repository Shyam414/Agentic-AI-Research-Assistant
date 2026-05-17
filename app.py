import uuid

import streamlit as st
from dotenv import load_dotenv

from tools.memory_store import build_semantic_memory_context, store_chat_memory
from workflow.graph import app

load_dotenv()

st.set_page_config(page_title="Agentic AI Research Assistant", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Agentic AI Research Assistant")

with st.sidebar:
    st.header("Session Memory")
    if st.session_state.history:
        for item in st.session_state.history[-5:]:
            st.caption(item)
    else:
        st.caption("No completed research yet.")

    if st.button("Clear Memory"):
        st.session_state.history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

query = st.text_input("Enter research topic", placeholder="Future of Agentic AI")

run = st.button("Run Agents", type="primary", disabled=not query.strip())

if run:
    recent_memory = "\n".join(st.session_state.history[-5:])
    semantic_memory = build_semantic_memory_context(query.strip())
    memory = "\n\n".join(item for item in [recent_memory, semantic_memory] if item)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Agents working..."):
        result = app.invoke({"topic": query.strip(), "memory": memory}, config=config)

    st.session_state.history.append(f"Topic: {query.strip()}\nSummary: {result['summary'][:500]}")
    try:
        store_chat_memory(
            query=query.strip(),
            summary=result["summary"],
            report=result["report"],
        )
    except Exception as error:
        st.warning(f"Could not save semantic memory: {error}")

    plan_tab, research_tab, summary_tab, report_tab = st.tabs(
        ["Research Plan", "Research Data", "Summary", "Final Report"]
    )

    with plan_tab:
        st.markdown(result["plan"])

    with research_tab:
        st.markdown(result["research"])

    with summary_tab:
        st.markdown(result["summary"])

    with report_tab:
        st.markdown(result["report"])
