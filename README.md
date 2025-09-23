# LangGraph

This repository contains a small LangGraph-based chatbot project. It includes a simple session manager and an advanced chatbot example.

## Files
- `langgraph_session.py` - session management utilities
- `langgraph_advance_chatbot.py` - example advanced chatbot runner
 - `advance_it_helpdesk_chatbot.py` - multi-agent IT helpdesk scaffold using LangGraph, Tools, and CosmosDB

## Quick start

1. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set environment variables (for example, OpenAI API key):

Create a `.env` file in the repo root with:

```
OPENAI_API_KEY=your_api_key_here
```

4. Run the example chatbot:

```powershell
python langgraph_advance_chatbot.py
```

### Advanced IT Helpdesk scaffold

There is a scaffold file `advance_it_helpdesk_chatbot.py` that demonstrates a multi-agent orchestration pattern with simple Tool adapters and optional Azure Cosmos DB persistence.

To use it, set the following environment variables (or add them to `.env`):

```
COSMOS_URL=https://<your-account>.documents.azure.com:443/
COSMOS_KEY=<your-primary-key>
OPENAI_API_KEY=<your-openai-key>
```

Then run:

```powershell
python advance_it_helpdesk_chatbot.py
```

The script is a scaffold — replace placeholder tool implementations and connect real LLM calls before using in production.
```

## Notes
- Update `requirements.txt` to match the exact versions you need.
- The included `.gitignore` hides common Python, Windows, and VS Code artifacts.
