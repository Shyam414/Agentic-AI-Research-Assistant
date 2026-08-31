# Agentic AI Research Assistant

A Streamlit research app that uses LangGraph, a local Ollama model, and Tavily web search. Enter a topic and the app creates a research plan, gathers web context, summarizes it, and writes a final report.

## Workflow

```text
planner -> researcher -> summarizer -> writer -> END
```

- **Planner** creates focused research areas, questions, technologies, and search queries.
- **Researcher** searches Tavily and extracts supported facts, trends, opportunities, risks, and open questions.
- **Summarizer** condenses the research into findings, evidence, sources, and uncertainties.
- **Writer** produces a professional report from the summary.

The planner requests Ollama JSON mode and also handles common local-model variations such as fenced JSON or a nested `research_plan` object. If the model omits search queries, topic-based fallback queries are used so the workflow can continue.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- A local chat model and embedding model (defaults shown below)
- A Tavily API key for live research
- Optional: MongoDB Atlas for persistent semantic memory

Pull the default models before starting the app:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text:v1.5
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create or update `.env`:

```env
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5
OLLAMA_MAX_NEW_TOKENS=1024
OLLAMA_MAX_RETRIES=2
TAVILY_API_KEY=your_tavily_api_key

# Optional semantic memory
MONGO_URI=your_mongodb_atlas_uri
MONGO_DB_NAME=agentic_research
MONGO_MEMORY_COLLECTION=chat_memories
MONGO_VECTOR_INDEX=chat_query_vector_index

3. Start the app:

```bash
streamlit run app.py
```

## Using the App

1. Enter a research topic.
2. Click **Run Agents**.
3. Review the Research Plan, Research Data, Summary, and Final Report tabs.
4. Use **Clear Memory** to reset only the current Streamlit session.

Without MongoDB or Redis, the app still runs; it displays a semantic-memory availability notice and continues with the research workflow. Tavily is required for successful live research.

## MongoDB Vector Search

For efficient semantic-memory retrieval, create an Atlas Vector Search index named `chat_query_vector_index` (or set `MONGO_VECTOR_INDEX` to its name):

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

If that index does not exist yet, the app falls back to cosine ranking over stored memories.

## Example Topics

- Future of Agentic AI
- AI in Healthcare
- Autonomous Coding Agents
- Multi-Agent Systems
- AI Automation in Finance
