# Agentic AI Research Assistant

A Streamlit research app built with LangGraph, Groq-hosted language models (GROQ_TOKEN / GROQ_MODEL_ID), and Tavily web search. A user submits a topic, the workflow creates a research plan, gathers web context, summarizes the findings, and produces a final report in separate UI tabs.

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
- `groq`
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
- A Groq API token with access to the selected model
- A Tavily API key for live web research

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create or update `.env`:

```env
GROQ_TOKEN=your_groq_api_token
GROQ_MODEL_ID=llama-3.1-8b-instant
GROQ_EMBEDDING_MODEL=nomic-embed-text-v1.5
TAVILY_API_KEY=your_tavily_api_key
MONGO_URI=your_mongodb_atlas_uri
MONGO_DB_NAME=agentic_research
MONGO_MEMORY_COLLECTION=chat_memories
MONGO_VECTOR_INDEX=chat_query_vector_index
REDIS_URL=redis://localhost:6379/0
REDIS_EMBEDDING_TTL_SECONDS=2592000
REDIS_RETRIEVAL_TTL_SECONDS=600
```

`GROQ_MODEL_ID` is optional. If omitted, the app defaults to `llama-3.1-8b-instant`.

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

- `GROQ_TOKEN` (or `GROQ_API_KEY`) is required. The current implementation raises an error if it is missing.
- `TAVILY_API_KEY` is optional for app startup, but research quality degrades without it. If missing, the researcher receives a placeholder message instead of live search results.
- The app uses Groq for chat completions and embeddings.
- `MONGO_URI` enables persistent semantic memory. Previous queries are embedded with `GROQ_EMBEDDING_MODEL`, stored in MongoDB, retrieved by vector similarity, and fed into the planner as related context.
- `REDIS_URL` enables optional caching. Embeddings and exact-query retrieval results are cached in Redis; if Redis is unavailable, the app falls back to Groq and MongoDB.

## MongoDB Vector Search

For fastest retrieval, create a MongoDB Atlas Vector Search index on the `chat_memories` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

Name the index `chat_query_vector_index`, or update `MONGO_VECTOR_INDEX` to match your index name.

If the vector index is not available yet, the app falls back to local cosine ranking over recent stored memories.

## Redis Cache

Redis is used only as a speed layer:

- Embedding cache keys avoid repeated Groq embedding calls for the same text.
- Retrieval cache keys avoid repeated MongoDB vector searches for the same query.
- Retrieval cache is cleared after storing a new chat memory, so newly saved queries can be retrieved immediately.

For local development, start Redis on `localhost:6379` and keep:

```env
REDIS_URL=redis://localhost:6379/0
```

## Example Topics

- Future of Agentic AI
- AI in Healthcare
- Autonomous Coding Agents
- Multi-Agent Systems
- AI Automation in Finance
