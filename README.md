# RAGentX

RAGentX is an **agentic** Retrieval-Augmented Generation system that actively searches the live web, retrieves fresh context, and reasons over it in real time to produce grounded, verifiable answers beyond static knowledge bases.[page:1]

## Features

- Agentic orchestration of multiple tools (search, retrieval, reasoning) instead of a single monolithic RAG call.[page:1]
- Live web search to fetch fresh, time-sensitive context rather than relying only on offline corpora.[page:1]
- Retrieval-augmented answer generation with explicit grounding in the retrieved evidence.
- Modular Python backend designed to be extended with new tools, agents, and data sources.[page:1]

## Project Structure

- `Backend/`: Core backend logic for agents, tools, and retrieval pipelines.[page:1]
- `main.py`: Entry point for running the RAGentX backend or launching example flows.[page:1]
- `requirements.txt`: Python dependencies required to run the project.[page:1]
- `pyproject.toml`: Project metadata and build configuration.[page:1]


