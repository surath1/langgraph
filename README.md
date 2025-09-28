# LangGraph examples and demos

This repository is a collection of small example projects and demos that demonstrate LangGraph patterns: flows, agents, tools, simple session management, and example front-ends.

The examples are intentionally small and meant for learning and prototyping. Many include placeholder tool implementations and stubbed LLM calls — inspect the source before using real API keys.

## Repository layout

- `Langgraph_basic/` — step-by-step examples (linear chains, conditional routing, agents, error handling). Files: `01_Simple_Linear_Chain.py` … `08_Customer_Service_Bot.py`.
- `Langgraph_chatbot/` — advanced chatbot examples and session utilities: `langgraph_advance_chatbot.py`, `langgraph_session.py`, `advance_it_helpdesk_chatbot.py`, and helper scripts.
- `Langgraph_multiagent_travel/` — multi-agent travel demo with a small backend, tools (currency, weather, place search), and a Streamlit front-end. See `README.md` inside that folder for front-end/back-end run steps.
- `requirements.txt` — top-level consolidated requirements (use folder-level requirements when provided).

Each example folder may include its own `requirements.txt`, `.env` examples, and a small README. Prefer using the folder-level instructions for that example.

## Quick getting-started (Windows cmd)

1) Create and activate a virtual environment in the workspace root:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies. For a project-wide install (quick, may install optional extras):

```cmd
pip install -r requirements.txt
```

Or install requirements for a specific example (recommended):

```cmd
cd Langgraph_multiagent_travel
pip install -r requirements.txt
```

3) Create a `.env` file inside the example folder (if required) and add your secrets. Typical variables used across examples:

```text
OPENAI_API_KEY=your_openai_api_key
COSMOS_URL=https://<your-account>.documents.azure.com:443/
COSMOS_KEY=<your-cosmos-key>
```

4) Run the example from its folder. Examples:

```cmd
cd Langgraph_chatbot
python langgraph_advance_chatbot.py
```

Streamlit front-end (multiagent travel)

The multi-agent travel demo has a small FastAPI backend and a Streamlit frontend. Below are recommended local steps for Windows (cmd.exe) to run the demo and a short reference for the backend endpoints used by the frontend.

1) Prepare a virtual environment inside the example folder and install the backend requirements:

```cmd
cd Langgraph_multiagent_travel
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

2) Start the FastAPI backend (from the example folder):

```cmd
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

3) In a separate terminal (activate the same venv), start the Streamlit frontend:

```cmd
cd Langgraph_multiagent_travel
.\.venv\Scripts\activate
streamlit run frontend/streamlit_app.py
```

Backend endpoints (the frontend talks to these under the `/api` prefix):

- GET  /api/health         -> health check
- POST /api/create_session -> returns a new session id
- POST /api/chat          -> main chat endpoint; request JSON: { "message": "..." } -> response JSON: { "answer": "<markdown>" }
- POST /api/chat_weather  -> chat endpoint that enables weather tool
- GET  /api/models        -> lists available model names (placeholder)
- POST /api/save_document -> saves a travel document payload (demo helper)

Example request for the main chat endpoint:

```json
{ "message": "Plan a 5-day trip to Bali for a family of 3, budget-friendly" }
```

If the frontend fails to get a response, confirm the backend is running at http://localhost:8000 and that `BASE_URL` in `frontend/streamlit_app.py` is set to `http://localhost:8000/api` (or adjust it to your host/port).

Docker notes

- Several folders include a `Dockerfile` (for example `Langgraph_chatbot` and `Langgraph_multiagent_travel`). If you prefer containers, build and run the Docker image from the folder with the Dockerfile. Check folder README for environment variables that need to be passed to the container at runtime.

Security and secrets

- Never commit secrets or API keys. Use a `.env` file (ignored by git) or your platform's secret manager. The code often reads `OPENAI_API_KEY` and Azure Cosmos environment variables — set these before running examples that need them.

Per-folder quick pointers

- `Langgraph_basic/`: Start with `01_Simple_Linear_Chain.py` and step through to `08_Customer_Service_Bot.py` to learn basic building blocks.
- `Langgraph_chatbot/`: `langgraph_session.py` contains session utilities. `langgraph_advance_chatbot.py` and `langgraph_helpdesk_cosmos.py` show integration examples.
- `Langgraph_multiagent_travel/`: A multi-agent demo with backend tools. See `Langgraph_multiagent_travel/README.md` for instructions to run the backend and Streamlit frontend.

Troubleshooting and tips

- If you get import errors, ensure the active virtual environment is activated and that you installed the correct `requirements.txt`.
- Inspect example code for placeholder functions before sending real requests to LLM providers.

Contributing

- Add or update folder-level README files when you change examples.
- Keep dependencies pinned in folder-level `requirements.txt` for reproducibility.

Need help running an example?

Tell me which example you want to run (for example: `Langgraph_chatbot/langgraph_advance_chatbot.py` or `Langgraph_multiagent_travel/frontend/streamlit_app.py`). I can open the files, install dependencies into a virtual environment, and run a quick smoke test.

---

