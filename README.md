# Agentic AI Research Assistant

A Streamlit research app built with LangGraph, Groq-hosted language models, and Tavily web search. A user submits a topic, the workflow creates a research plan, gathers web context, summarizes the findings, and produces a final report in separate UI tabs.

## What It Does

- Accepts a research topic from the Streamlit interface
- Maintains lightweight session memory for recent completed topics
- Runs a four-stage LangGraph pipeline:
  - Planner
  - Researcher
  - Summarizer
  - Writer
- Uses Tavily search results as research input
- Produces:
  - Research plan
  - Research data
  - Summary
  - Final report

## Workflow

The graph in `workflow/graph.py` executes in this order:

```text
planner -> researcher -> summarizer -> writer -> END
```

Each step enriches the shared workflow state:

- `planner`: builds a focused research plan from the topic and recent session context
- `researcher`: runs Tavily search and extracts facts, trends, opportunities, risks, and open questions
- `summarizer`: condenses research into concise findings and uncertainties
- `writer`: turns the summary into a professional report

LangGraph is compiled with an in-memory checkpointer, and the Streamlit app passes a per-session `thread_id`.

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain
- `langchain-groq`
- Tavily Search API
- `python-dotenv`

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- agents/
|   |-- llm.py
|   |-- planner.py
|   |-- researcher.py
|   |-- summarizer.py
|   `-- writer.py
|-- tools/
|   `-- search_tool.py
`-- workflow/
    `-- graph.py
```

## Requirements

- Python 3.10+ recommended
- A Groq API key
- A Tavily API key for live web research

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create or update `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_tavily_api_key
```

`GROQ_MODEL` is optional. If omitted, the app defaults to `llama-3.3-70b-versatile`.

3. Start the app:

```bash
streamlit run app.py
```

## Using the App

1. Enter a topic in the text field.
2. Click `Run Agents`.
3. Review the generated tabs:
   - `Research Plan`
   - `Research Data`
   - `Summary`
   - `Final Report`
4. Use `Clear Memory` in the sidebar to reset the recent-session context and create a fresh workflow thread.

## Environment Notes

- `GROQ_API_KEY` is required. The current implementation raises an error if it is missing.
- `TAVILY_API_KEY` is optional for app startup, but research quality degrades without it. If missing, the researcher receives a placeholder message instead of live search results.
- The current code uses Groq only. It does not currently instantiate OpenAI models.

## Example Topics

- Future of Agentic AI
- AI in Healthcare
- Autonomous Coding Agents
- Multi-Agent Systems
- AI Automation in Finance
