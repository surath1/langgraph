# LangGraph

This repository contains a small LangGraph-based projects. It includes a simple session manager and an advanced chatbot example.
# LangGraph workspace

This repository is a collection of small example projects and demos that use LangGraph patterns (flows, agents, tools, and simple session management). Each subfolder is a separate example you can explore. The goal of this top-level README is to help new contributors quickly discover and run the examples on Windows.

## Repository layout

- `Langgraph_100/` — Docker-based example and simple runner (`langgraph_001.py`).
- `Langgraph_101/` — another example project with a `Dockerfile` and `langgraph_101.py`.
- `Langgraph_102/` — several example scripts including `langgraph_advance_chatbot.py`, `advance_it_helpdesk_chatbot.py`, and `langgraph_session.py` (session management and advanced chatbot examples).
- `Langgraph_basic/` — small step-by-step examples (linear chains, conditional routing, agents, error handling).
- `Langgraph_multiagent_travel/` — multi-agent travel demo with a small backend, tools, and a Streamlit front-end. See `streamlit_app.py` and `backend/` for components.

Each folder contains its own `README.md` or notes and a `requirements.txt` where applicable. Before running any example, check the folder-level README for project-specific instructions.

## Quick getting-started (Windows cmd)

1. Create and activate a virtual environment in the workspace root (cmd):

```
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies for the example you want to run. For a top-level install you can run:

```
pip install -r requirements.txt
```

Or install the requirements for a specific example, for example the multiagent travel demo:

```
cd Langgraph_multiagent_travel
pip install -r requirements.txt
```

3. Create a `.env` file in the folder containing the example (if required) and add any secrets. Example variables commonly used across examples:

```
OPENAI_API_KEY=your_openai_api_key
COSMOS_URL=https://<your-account>.documents.azure.com:443/
COSMOS_KEY=<your-cosmos-key>
```

4. Run the example script from its folder. Examples:

```
cd Langgraph_102
python langgraph_advance_chatbot.py
```

Or for the multiagent travel Streamlit app:

```
cd Langgraph_multiagent_travel
streamlit run streamlit_app.py
```

Notes:
- Many examples are scaffolds and include placeholder tool implementations or stubbed LLM calls. Inspect the source before using with real API keys.
- If a folder provides a `Dockerfile` or `docker-compose.yml` you can run the containerized example instead of installing locally.

## Per-project quick pointers

- Langgraph_basic/: Browse `01_Simple_Linear_Chain.py` through `08_Customer_Service_Bot.py` to see progressively more advanced patterns.
- Langgraph_102/: `langgraph_session.py` contains session utilities. `langgraph_advance_chatbot.py` is a runnable advanced chatbot example. `advance_it_helpdesk_chatbot.py` is a multi-agent scaffold.
- Langgraph_multiagent_travel/: The backend contains tools (currency, weather, place search) and `endpoints/main.py` exposes APIs used by the frontend. Start with `README.md` in that folder.

## Contributing

- If you add or update an example, please include or update the folder-level `README.md` with run steps and dependency versions.
- Keep secrets out of the repository. Use `.env` and your platform's secret manager for production.

## Next steps

1. Open a folder you're interested in (for example `Langgraph_102`) and run its `README.md` steps.
2. If you want, tell me which example to prepare or run and I can open the file, install dependencies, and run a quick smoke test.

---

Updated README to list project folders and give quick Windows cmd instructions for exploring the examples.
