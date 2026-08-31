import uuid
import streamlit as st
from dotenv import load_dotenv
from tools.memory_store import (
    build_semantic_memory_context,
    store_chat_memory,
)
from workflow.graph import app

load_dotenv()

st.set_page_config(
    page_title="Agentic AI Research Assistant",
    layout="wide",
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.title("Agentic AI Research Assistant")
with st.sidebar:
    st.header("Session")

    st.caption(f"Thread ID: {st.session_state.thread_id}")

    if st.button("New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

query = st.text_input("Enter research topic",placeholder="Future of Agentic AI",)
run = st.button("Run Agents",type="primary",disabled=not query.strip(),)

if run:
    current_query = query.strip()
    try:
        semantic_memory = build_semantic_memory_context(current_query)
    except Exception as error:
        semantic_memory = ""
        st.warning(f"Could not retrieve semantic memory: {error}")

    memory = semantic_memory.strip() if semantic_memory else ""
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.spinner("Agents working..."):

        result = app.invoke(
            {
                "topic": current_query,
                "memory": memory,
            },
            config=config,
        )

    try:
        store_chat_memory(
            query=current_query,
            summary=result["summary"],
            report=result["report"],
        )

    except Exception as error:
        st.warning(f"Could not save semantic memory: {error}")

    plan_tab, research_tab, summary_tab, report_tab = st.tabs(
        [
            "Research Plan",
            "Research Data",
            "Summary",
            "Final Report",
        ]
    )

    with plan_tab:

        plan_labels = {
            "research_areas": "Research areas",
            "questions": "Key questions",
            "technologies": "Relevant technologies and entities",
            "search_queries": "Search queries",
        }

        for key, label in plan_labels.items():

            items = result["plan"].get(key, [])

            if items:
                st.subheader(label)

                st.markdown(
                    "\n".join(
                        f"- {item}"
                        for item in items
                    )
                )

    with research_tab:
        st.markdown(result["research"])

    with summary_tab:
        st.markdown(result["summary"])

    with report_tab:
        st.markdown(result["report"])