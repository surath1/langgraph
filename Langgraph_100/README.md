# ==========================================
# README.md
# ==========================================
# IT Service Desk ChatBot

A simple yet powerful IT Service Desk ChatBot built with:
- **LangGraph**: 2-agent workflow (Orchestrator + Executor) 
- **FastAPI**: REST API with automatic documentation
- **CosmosDB**: Session storage and user history
- **ReAct Agent**: Tool-enabled executor with reasoning

## 🏗️ Architecture

```
User Request → FastAPI → LangGraph Workflow
                            ↓
                      [Orchestrator Agent] 
                      (Analyzes & Classifies)
                            ↓
                      [Executor Agent]
                      (ReAct + Tools)
                            ↓
                        CosmosDB
                      (Save Session)
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

4. **Test the API**
   - Open: http://localhost:8000/docs
   - Or run: `python test_requests.py`

## 📋 Supported Request Types

- **Password Reset**: "I forgot my password"
- **Software Access**: "I need access to Adobe Photoshop"  
- **Hardware Issues**: "My laptop screen is broken"
- **General Support**: "How do I connect to VPN?"

## 🔧 Tools Available

- `password_reset_tool`: Sends password reset emails
- `software_access_tool`: Creates software access requests
- `hardware_support_tool`: Provides hardware troubleshooting
- `create_general_ticket_tool`: Creates support tickets

## 🛠️ Configuration

Update the `Config` class in `main.py`:
- CosmosDB endpoint and key
- OpenAI API key
- Debug settings

## 📚 API Endpoints

- `POST /chat`: Main chat interface
- `GET /user/{user_id}/history`: User chat history
- `GET /session/{session_id}`: Session details
- `GET /health`: Health check

## 🧪 Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I need a password reset", "user_id": "test_user"}'
```

## 🔄 Workflow Flow

1. **User sends message** → FastAPI endpoint
2. **Orchestrator Agent** → Classifies request type
3. **Executor Agent** → Uses ReAct pattern with tools
4. **CosmosDB** → Saves session data
5. **Response** → Returns to user

Simple, effective, and easy to extend! 🎉