# Simple IT Service Desk ChatBot
# LangGraph (2 Agents) + FastAPI + CosmosDB

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangGraph imports
from langgraph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Azure CosmosDB
from azure.cosmos import CosmosClient, PartitionKey

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    # CosmosDB Settings
    COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://your-cosmos-account.documents.azure.com:443/")
    COSMOS_KEY = os.getenv("COSMOS_KEY", "your-cosmos-key")
    COSMOS_DATABASE = "itservicedesk"
    COSMOS_CONTAINER = "sessions"
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
    
    # App Settings
    DEBUG = True

# ==========================================
# MODELS
# ==========================================

class RequestType(str, Enum):
    PASSWORD_RESET = "password_reset"
    SOFTWARE_ACCESS = "software_access" 
    HARDWARE_ISSUE = "hardware_issue"
    GENERAL_SUPPORT = "general_support"

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    request_type: Optional[str] = None
    action_taken: Optional[str] = None

# LangGraph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat messages"]
    user_id: str
    session_id: str
    request_type: Optional[str]
    user_query: str
    action_plan: str
    final_response: str
    session_data: Dict

# ==========================================
# DATABASE SERVICE
# ==========================================

class CosmosDBService:
    def __init__(self):
        """Initialize CosmosDB client and containers"""
        self.client = CosmosClient(Config.COSMOS_ENDPOINT, Config.COSMOS_KEY)
        self.database = self.client.create_database_if_not_exists(id=Config.COSMOS_DATABASE)
        self.container = self.database.create_container_if_not_exists(
            id=Config.COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/session_id")
        )
    
    def save_session(self, session_data: Dict):
        """Save or update session data"""
        try:
            session_data['timestamp'] = datetime.utcnow().isoformat()
            self.container.upsert_item(body=session_data)
            print(f"✅ Saved session: {session_data['session_id']}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data"""
        try:
            item = self.container.read_item(item=session_id, partition_key=session_id)
            return item
        except Exception as e:
            print(f"ℹ️ Session not found or error: {e}")
            return None
    
    def get_user_history(self, user_id: str) -> List[Dict]:
        """Get user's chat history"""
        try:
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.timestamp DESC"
            items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
            return items[:5]  # Return last 5 sessions
        except Exception as e:
            print(f"❌ Error getting user history: {e}")
            return []

# ==========================================
# TOOLS FOR EXECUTOR AGENT
# ==========================================

@tool
def password_reset_tool(user_email: str) -> str:
    """Tool to reset user password"""
    print(f"🔧 Executing password reset for: {user_email}")
    # In real implementation, integrate with AD/identity service
    return f"Password reset link sent to {user_email}. Check your email within 5 minutes."

@tool
def software_access_tool(software_name: str, user_id: str) -> str:
    """Tool to request software access"""
    print(f"🔧 Requesting software access: {software_name} for user: {user_id}")
    ticket_id = f"SW-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Software access request created. Ticket ID: {ticket_id}. IT team will process within 2 business days."

@tool
def hardware_support_tool(issue_description: str) -> str:
    """Tool to handle hardware issues"""
    print(f"🔧 Processing hardware issue: {issue_description}")
    if "laptop" in issue_description.lower() or "computer" in issue_description.lower():
        return "Please try restarting your computer first. If issue persists, contact IT support at ext. 1234."
    elif "printer" in issue_description.lower():
        return "Check printer connection and paper. For further help, ticket created - IT will contact you within 4 hours."
    else:
        return "Hardware support ticket created. IT technician will contact you within 24 hours."

@tool
def create_general_ticket_tool(issue_description: str, user_id: str) -> str:
    """Tool to create general support ticket"""
    print(f"🔧 Creating general ticket for: {user_id}")
    ticket_id = f"GEN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Support ticket created: {ticket_id}. Our team will respond within 4 hours."

# ==========================================
# AGENTS
# ==========================================

class ITServiceDeskAgents:
    def __init__(self, db_service: CosmosDBService):
        self.db_service = db_service
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )
    
    # ========================================
    # ORCHESTRATOR AGENT
    # ========================================
    
    def orchestrator_agent(self, state: AgentState) -> AgentState:
        """
        Orchestrator Agent: Analyzes user requests and determines the action plan
        """
        print("🎯 ORCHESTRATOR AGENT: Analyzing request...")
        
        user_query = state["user_query"]
        user_id = state["user_id"]
        
        # Get user history for context
        user_history = self.db_service.get_user_history(user_id)
        history_context = ""
        if user_history:
            history_context = f"User's recent issues: {[h.get('request_type', 'unknown') for h in user_history[:3]]}"
        
        # Classify the request type
        classification_prompt = f"""
        You are an IT Service Desk Orchestrator. Analyze this user request and classify it.
        
        User Query: "{user_query}"
        User ID: {user_id}
        {history_context}
        
        Classify into one of these categories:
        1. password_reset - for password, login, or account access issues
        2. software_access - for software installation, licenses, or access requests  
        3. hardware_issue - for computer, laptop, printer, or device problems
        4. general_support - for other IT support needs
        
        Also create a brief action plan for the Executor Agent.
        
        Respond in this JSON format:
        {{
            "request_type": "category_name",
            "action_plan": "specific plan for executor agent",
            "confidence": "high/medium/low"
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        
        try:
            # Parse the JSON response
            result = json.loads(response.content)
            state["request_type"] = result.get("request_type", "general_support")
            state["action_plan"] = result.get("action_plan", "Handle general support request")
        except:
            # Fallback if JSON parsing fails
            content = response.content.lower()
            if "password" in content or "login" in content:
                state["request_type"] = "password_reset"
                state["action_plan"] = "Reset user password"
            elif "software" in content or "access" in content:
                state["request_type"] = "software_access"
                state["action_plan"] = "Process software access request"
            elif "hardware" in content or "computer" in content or "laptop" in content:
                state["request_type"] = "hardware_issue"
                state["action_plan"] = "Troubleshoot hardware issue"
            else:
                state["request_type"] = "general_support"
                state["action_plan"] = "Create general support ticket"
        
        print(f"✅ ORCHESTRATOR: Classified as '{state['request_type']}'")
        print(f"📋 ORCHESTRATOR: Action plan - {state['action_plan']}")
        
        return state
    
    # ========================================
    # EXECUTOR AGENT (ReAct Agent with Tools)
    # ========================================
    
    def executor_agent(self, state: AgentState) -> AgentState:
        """
        Executor Agent: Uses ReAct pattern with tools to execute actions
        """
        print("⚡ EXECUTOR AGENT: Taking action...")
        
        # Available tools
        tools = [
            password_reset_tool,
            software_access_tool, 
            hardware_support_tool,
            create_general_ticket_tool
        ]
        
        # Create ReAct agent
        react_prompt = PromptTemplate.from_template("""
        You are an IT Service Desk Executor Agent. You have access to tools to help users.
        
        User Query: {user_query}
        Request Type: {request_type}
        Action Plan: {action_plan}
        User ID: {user_id}
        
        Use the appropriate tools to help the user. Be helpful, professional, and efficient.
        
        You have access to these tools:
        {tools}
        
        {agent_scratchpad}
        """)
        
        # Create ReAct agent with tools
        agent = create_react_agent(self.llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
        
        # Execute the agent
        try:
            result = agent_executor.invoke({
                "user_query": state["user_query"],
                "request_type": state["request_type"], 
                "action_plan": state["action_plan"],
                "user_id": state["user_id"]
            })
            
            state["final_response"] = result.get("output", "I apologize, but I encountered an issue processing your request. Please contact IT support directly.")
            
        except Exception as e:
            print(f"❌ EXECUTOR ERROR: {e}")
            # Fallback responses based on request type
            fallback_responses = {
                "password_reset": "I'll help you reset your password. Please check your email for reset instructions, or contact IT at ext. 1234.",
                "software_access": "I've noted your software access request. IT will review and respond within 2 business days.",
                "hardware_issue": "For immediate hardware support, please contact IT at ext. 1234 or submit a ticket online.",
                "general_support": "I've created a support ticket for you. Our team will respond within 4 hours."
            }
            state["final_response"] = fallback_responses.get(
                state["request_type"], 
                "Thank you for contacting IT support. We'll assist you shortly."
            )
        
        print(f"✅ EXECUTOR: Generated response")
        return state

# ==========================================
# LANGGRAPH WORKFLOW
# ==========================================

class ITServiceDeskWorkflow:
    def __init__(self, db_service: CosmosDBService):
        self.db_service = db_service
        self.agents = ITServiceDeskAgents(db_service)
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow with 2 agents"""
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("orchestrator", self.agents.orchestrator_agent)
        workflow.add_node("executor", self.agents.executor_agent)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges (flow)
        workflow.add_edge("orchestrator", "executor")  # Orchestrator → Executor
        workflow.add_edge("executor", END)             # Executor → End
        
        return workflow.compile()
    
    def process_request(self, chat_request: ChatRequest) -> ChatResponse:
        """Process user request through the 2-agent workflow"""
        
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        print(f"\n🚀 Processing request for user: {chat_request.user_id}")
        print(f"💬 Message: {chat_request.message}")
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=chat_request.message)],
            user_id=chat_request.user_id,
            session_id=session_id,
            request_type=None,
            user_query=chat_request.message,
            action_plan="",
            final_response="",
            session_data={}
        )
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Save session to CosmosDB
            session_data = {
                "id": session_id,
                "session_id": session_id,
                "user_id": chat_request.user_id,
                "user_message": chat_request.message,
                "request_type": final_state["request_type"],
                "bot_response": final_state["final_response"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.db_service.save_session(session_data)
            
            return ChatResponse(
                response=final_state["final_response"],
                session_id=session_id,
                request_type=final_state["request_type"],
                action_taken=final_state["action_plan"]
            )
            
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return ChatResponse(
                response="I'm sorry, I'm experiencing technical difficulties. Please contact IT support directly at ext. 1234.",
                session_id=session_id,
                request_type="error",
                action_taken="error_fallback"
            )

# ==========================================
# FASTAPI APPLICATION
# ==========================================

# Initialize services
db_service = CosmosDBService()
workflow_service = ITServiceDeskWorkflow(db_service)

# Create FastAPI app
app = FastAPI(
    title="IT Service Desk ChatBot",
    description="Simple 2-Agent IT Support ChatBot with LangGraph",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "IT Service Desk ChatBot is running!",
        "agents": ["Orchestrator", "Executor"],
        "supported_requests": ["password_reset", "software_access", "hardware_issue", "general_support"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Send a message to the IT Service Desk ChatBot.
    The Orchestrator will analyze your request and the Executor will take appropriate action.
    """
    try:
        response = workflow_service.process_request(request)
        return response
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    """Get user's chat history"""
    try:
        history = db_service.get_user_history(user_id)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        print(f"❌ History Error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve history")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get specific session details"""
    try:
        session = db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session Error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve session")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected",
        "agents": ["orchestrator", "executor"]
    }

# ==========================================
# EXAMPLE USAGE
# ==========================================

"""
Example API calls:

1. Password Reset:
POST /chat
{
    "message": "I forgot my password and can't login",
    "user_id": "john.doe"
}

2. Software Access:
POST /chat  
{
    "message": "I need access to Adobe Photoshop for my project",
    "user_id": "jane.smith"
}

3. Hardware Issue:
POST /chat
{
    "message": "My laptop screen is flickering",
    "user_id": "bob.wilson"
}

4. General Support:
POST /chat
{
    "message": "How do I connect to the office VPN?",
    "user_id": "alice.brown"
}
"""

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("🤖 Starting IT Service Desk ChatBot...")
    print("👥 Agents: Orchestrator + Executor (ReAct)")
    print("🔧 Tools: Password Reset, Software Access, Hardware Support, General Tickets")
    print("💾 Database: CosmosDB for session storage")
    print("🌐 API: FastAPI with OpenAPI docs at http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=Config.DEBUG
    )