
# Travel Planner — Agentic Application (Langgraph Multi-Agent)

An experimental agentic travel-planning application built with LangGraph, LangChain integrations, and several small tooling components (weather, place search, currency conversion, expense calculator). The app exposes a FastAPI backend that constructs a LangGraph workflow (LLM + tools) and a Streamlit frontend that sends user queries and displays AI-generated travel plans.

This repository is intended as a reference implementation and research prototype. It demonstrates how to wire an LLM, a graph-based workflow, and a set of tools into an interactive travel planning assistant.

## Table of contents

- Project overview
- Quickstart (run locally)
- Configuration
- Architecture and important modules
- Development notes
- File map
- Troubleshooting

## Project overview

The system consists of:

- A FastAPI backend (`endpoints/main.py`) that builds the workflow graph using `backend.agent.workflow.GraphBuilder`. The backend exposes endpoints to create a session and query the agent.
- A Streamlit frontend (`streamlit_app.py`) that sends user inputs to the backend and renders the returned Markdown plan.
- A small tools collection under `backend/tools/` (weather, place search, currency conversion, expense calculator) used by the agent to fetch or compute data.
- LLM loader and configuration under `backend/utils/` and `backend/config/config.yaml` that allow switching between providers (OpenAI, Groq, Gemini, Ollama — with Groq/OpenAI supported in code).

Important: This project is a prototype. Several pieces (production-grade error handling, secret management, credentials validation, and rate limiting) should be hardened before using with real API keys.

## Quickstart (local development)

Prerequisites

- Python 3.10+ (create a venv recommended)
- Required Python packages listed in `requirements.txt` (install into your venv)
- API keys for the LLM provider you plan to use (e.g., `OPENAI_API_KEY` or `GROQ_API_KEY`) set in environment variables or a `.env` file.

Steps

1. Create and activate a virtual environment:

	- Windows (cmd.exe):

	  ```cmd
	  python -m venv .venv
	  .venv\Scripts\activate
	  pip install -r requirements.txt
	  ```

2. Create a `.env` (optional) in the project root and add your keys (example):

	```text
	OPENAI_API_KEY=sk-...
	GROQ_API_KEY=...
	```

3. Start the backend FastAPI server (from project root):

	```cmd
	uvicorn endpoints.main:app --host 0.0.0.0 --port 8000 --reload
	```

4. Start the Streamlit frontend (in a separate terminal):

	```cmd
	streamlit run streamlit_app.py
	```

5. Open the Streamlit UI (usually at http://localhost:8501) and enter a travel planning query (for example: "Plan a 5-day trip to Goa"). The app sends the query to `http://localhost:8000/app/query` and displays the generated Markdown plan.

API endpoints (from the backend)

- GET /app/health — health check
- POST /app/create_session — builds and saves a visual graph (saves `my_graph.png`)
- POST /app/query — accepts a JSON payload with `message` and returns `{"answer": "..."}`

Example query payloads

POST /app/query

Request body (JSON):

```json
{ "message": "Plan a 5-day trip to Bali for a family of 3, budget-friendly" }
```

Response (JSON):

```json
{ "answer": "<Markdown travel plan>" }
```

## Configuration

Core configuration resides in `backend/config/config.yaml`. This file contains provider-specific defaults and model names. The active model provider is selected in code when instantiating `GraphBuilder(model_provider=...)` or `ModelLoader(model_provider=...)`.

Environment variables used

- OPENAI_API_KEY — OpenAI API key (if using OpenAI provider)
- GROQ_API_KEY — Groq API key (if using Groq provider)
- Any other provider keys you wire into `backend/utils/llm_loader.py`

Note: The repo reads config via `backend/utils/config_loader.py` (helper) and `ModelLoader` uses the loaded config to pick model names and defaults.

## Architecture and important modules

- `endpoints/main.py` — FastAPI app that wires the graph builder and returns agent responses. Look here to change endpoints or add authentication.
- `streamlit_app.py` — Minimal Streamlit front-end used for quick testing and demo.
- `backend/agent/workflow.py` — `GraphBuilder` class: constructs a LangGraph `StateGraph` with an `agent` node (invokes LLM) and a `tools` node (ToolNode). This is where tool binding and LLM invocation logic live.
- `backend/utils/llm_loader.py` — `ModelLoader` and `ConfigLoader` that pick and instantiate specific LLM clients (OpenAI / Groq currently supported). Add provider logic here when needed.
- `backend/prompts/prompt.py` — System / human / AI example messages that seed the conversation.
- `backend/tools/` — Tool implementations that the agent may call. They expose small functions wrapped as tool nodes for the LLM.

Design contract (inputs/outputs)

- Input: user string message (JSON {"message": "..."})
- Output: JSON {"answer": "<markdown>"} where answer is the agent's Markdown-formatted travel plan.

Edge cases to consider

- Empty/invalid input: backend currently expects a non-empty `message` field. Add validation if needed.
- Missing API keys: the LLM loader will raise errors; ensure environment variables are provided.
- Long-running LLM calls: the configured timeouts/defaults live in `config.yaml`.

## Development notes

- To change the model provider, update `GraphBuilder(model_provider="openai"|"groq"|...)` or change defaults in `backend/config/config.yaml`.
- Add tests: there are no unit tests in the repository. Consider adding tests for the LLM loader and tool wrappers.
- Tooling: the repo uses environment variables for secrets. For production, integrate a secrets manager.

## File map (high level)

- `streamlit_app.py` — Simple UI
- `requirements.txt` — Python dependencies
- `endpoints/main.py` — FastAPI app and endpoints
- `backend/agent/workflow.py` — Graph builder and agent integration
- `backend/config/config.yaml` — Model/provider configuration
- `backend/prompts/prompt.py` — System/human/AI prompt templates
- `backend/tools/` — Set of small tools (weather, place search, currency conversion, expense calculator)
- `backend/utils/` — Helpers: `llm_loader.py`, `config_loader.py`, `document.py`, etc.

## Troubleshooting & Tips

- If the frontend shows "Bot failed to respond", inspect the backend logs and ensure the backend is reachable at `BASE_URL` (default http://localhost:8000 in `streamlit_app.py`).
- If the LLM provider errors, confirm the appropriate API key is set and that model names in `config.yaml` match the provider.
- If you see import errors for provider-specific packages (e.g., `langchain_groq`), install them or remove/replace provider logic you don't need.

## Next steps / Suggested improvements

- Add more robust input validation and typed request models for the `/app/query` endpoint.
- Add unit/integration tests for core components (ModelLoader, GraphBuilder, tools).
- Use structured logging and a per-request tracing id to make debugging multi-step agent flows easier.
- Add a Dockerfile and compose setup to simplify local development and multi-service runs.

## License

This repository is a prototype; no license file is included. Add a license if you plan to open-source or share.

---

If you'd like, I can also:

- Add a small Dockerfile and docker-compose for local dev
- Add a basic unit test for `backend/utils/llm_loader.py`
- Wire a minimal `.env.example` with the environment variables used
