# Multi-Agent Research Assistant using LangGraph and OpenAI

A Streamlit app that runs a LangGraph workflow with specialized agents for planning, web research, summarization, and report writing.

## Features

- Planner agent
- Web research agent using Tavily
- Summarizer agent
- Report writer agent
- Streamlit UI
- Basic session memory
- OpenAI by default, optional Groq support

## Setup

```bash
pip install -r requirements.txt
```

Update `.env` with your API keys:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

Run:

```bash
streamlit run app.py
```

## Example Topics

- Future of Agentic AI
- AI in Healthcare
- Autonomous Coding Agents
- Multi-Agent Systems
- AI Automation in Finance
