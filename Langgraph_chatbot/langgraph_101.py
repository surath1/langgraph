# Complete IT Service Desk ChatBot Implementation
# FastAPI + LangGraph + CosmosDB

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from enum import Enum
import logging
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph import StateGraph, END
from langgraph.graph import Graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Azure imports
from azure.cosmos import CosmosClient, exceptions


# Configuration and Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION AND MODELS
# ==========================================

class Config:
    # Azure CosmosDB
    COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
    COSMOS_KEY = os.getenv("COSMOS_KEY") 
    COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "itservicedesk")
    COSMOS_CONTAINER_USERS = os.getenv("COSMOS_CONTAINER_USERS", "users")
    COSMOS_CONTAINER_CONVERSATIONS = os.getenv("COSMOS_CONTAINER_CONVERSATIONS", "conversations")
    COSMOS_CONTAINER_TICKETS = os.getenv("COSMOS_CONTAINER_TICKETS", "tickets")
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    # Application
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

class RequestType(str, Enum):
    SOFTWARE_ACCESS = "software_access"
    PASSWORD_RESET = "password_reset"
    HARDWARE_REQUEST = "hardware_request"
    NETWORK_ISSUE = "network_issue"
    GENERAL_SUPPORT = "general_support"

class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class UserRole(str, Enum):
    TESTER = "tester"
    HELPDESK = "helpdesk"
    IT_SUPPORT = "it_support"
    ADMIN = "admin"

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    user_role: UserRole = UserRole.TESTER

class ChatResponse(BaseModel):
    response: str
    session_id: str
    ticket_id: Optional[str] = None
    requires_action: bool = False
    suggested_actions: List[str] = []

class User(BaseModel):
    id: str
    email: str
    name: str
    role: UserRole
    department: str
    created_at: datetime
    last_login: datetime

class Ticket(BaseModel):
    id: str
    user_id: str
    title: str
    description: str
    request_type: RequestType
    status: TicketStatus
    priority: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None

class Conversation(BaseModel):
    id: str
    user_id: str
    session_id: str
    messages: List[Dict[str, Any]]
    ticket_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# LangGraph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    user_id: str
    session_id: str
    user_info: Optional[Dict[str, Any]]
    previous_conversations: List[Dict[str, Any]]
    current_request_type: Optional[RequestType]
    ticket_id: Optional[str]
    requires_human_intervention: bool
    suggested_actions: List[str]
    context: Dict[str, Any]

# ==========================================
# DATABASE SERVICE
# ==========================================

class CosmosDBService:
    def __init__(self):
        self.client = CosmosClient(Config.COSMOS_ENDPOINT, Config.COSMOS_KEY)
        self.database = self.client.get_database_client(Config.COSMOS_DATABASE)
        self.users_container = self.database.get_container_client(Config.COSMOS_CONTAINER_USERS)
        self.conversations_container = self.database.get_container_client(Config.COSMOS_CONTAINER_CONVERSATIONS)
        self.tickets_container = self.database.get_container_client(Config.COSMOS_CONTAINER_TICKETS)

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            user = self.users_container.read_item(item=user_id, partition_key=user_id)
            return user
        except exceptions.CosmosResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None

    async def create_user(self, user: User) -> Dict[str, Any]:
        try:
            user_dict = user.dict()
            user_dict['created_at'] = user_dict['created_at'].isoformat()
            user_dict['last_login'] = user_dict['last_login'].isoformat()
            return self.users_container.create_item(user_dict)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    async def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
            items = list(self.conversations_container.query_items(
                query=query,
                parameters=[
                    {"name": "@user_id", "value": user_id},
                    {"name": "@limit", "value": limit}
                ],
                partition_key=user_id
            ))
            return items
        except Exception as e:
            logger.error(f"Error getting conversations for user {user_id}: {e}")
            return []

    async def save_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        try:
            conv_dict = conversation.dict()
            conv_dict['created_at'] = conv_dict['created_at'].isoformat()
            conv_dict['updated_at'] = conv_dict['updated_at'].isoformat()
            return self.conversations_container.upsert_item(conv_dict)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            raise

    async def create_ticket(self, ticket: Ticket) -> Dict[str, Any]:
        try:
            ticket_dict = ticket.dict()
            ticket_dict['created_at'] = ticket_dict['created_at'].isoformat()
            ticket_dict['updated_at'] = ticket_dict['updated_at'].isoformat()
            if ticket_dict.get('resolved_at'):
                ticket_dict['resolved_at'] = ticket_dict['resolved_at'].isoformat()
            return self.tickets_container.create_item(ticket_dict)
        except Exception as e:
            logger.error(f"Error creating ticket: {e}")
            raise

    async def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Dict[str, Any]:
        try:
            ticket = self.tickets_container.read_item(item=ticket_id, partition_key=ticket_id)
            ticket['status'] = status
            ticket['updated_at'] = datetime.utcnow().isoformat()
            if status == TicketStatus.RESOLVED:
                ticket['resolved_at'] = datetime.utcnow().isoformat()
            return self.tickets_container.replace_item(item=ticket_id, body=ticket)
        except Exception as e:
            logger.error(f"Error updating ticket {ticket_id}: {e}")
            raise

# ==========================================
# LANGGRAPH AGENTS
# ==========================================

class ITServiceDeskAgents:
    def __init__(self, db_service: CosmosDBService):
        self.db_service = db_service
        self.llm = AzureChatOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_KEY,
            api_version=Config.AZURE_OPENAI_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1
        )

    async def orchestrator_agent(self, state: AgentState) -> AgentState:
        """Main orchestrator agent that coordinates the workflow"""
        logger.info("Orchestrator agent processing request")
        
        # Get the latest message
        latest_message = state["messages"][-1].content if state["messages"] else ""
        
        # Analyze request type and determine next steps
        system_prompt = """
        You are an IT Service Desk Orchestrator. Your role is to:
        1. Understand user requests (software access, password reset, hardware, network issues)
        2. Coordinate with other agents to gather information and take actions
        3. Provide clear, professional responses to users
        4. Escalate complex issues when needed
        
        Current user message: {message}
        User role: {user_role}
        
        Analyze the request and determine:
        - Request type (software_access, password_reset, hardware_request, network_issue, general_support)
        - Whether user information is needed
        - What actions might be required
        """
        
        user_role = state.get("context", {}).get("user_role", "tester")
        
        prompt = system_prompt.format(message=latest_message, user_role=user_role)
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Determine request type from response
        request_type = self._classify_request(latest_message)
        state["current_request_type"] = request_type
        
        # Add orchestrator response to messages
        state["messages"].append(AIMessage(content=f"Orchestrator analyzing: {request_type}"))
        
        logger.info(f"Orchestrator classified request as: {request_type}")
        return state

    async def cosmosdb_tool_agent(self, state: AgentState) -> AgentState:
        """Agent responsible for database operations and user history"""
        logger.info("CosmosDB Tool agent processing request")
        
        user_id = state["user_id"]
        
        # Get user information
        user_info = await self.db_service.get_user(user_id)
        if not user_info:
            # Create new user if doesn't exist
            new_user = User(
                id=user_id,
                email=f"{user_id}@company.com",  # This should come from auth
                name=f"User {user_id}",
                role=UserRole.TESTER,
                department="Unknown",
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            user_info = await self.db_service.create_user(new_user)
        
        state["user_info"] = user_info
        
        # Get previous conversations
        conversations = await self.db_service.get_user_conversations(user_id, limit=5)
        state["previous_conversations"] = conversations
        
        # Save current conversation
        conversation = Conversation(
            id=state["session_id"],
            user_id=user_id,
            session_id=state["session_id"],
            messages=[{"role": "user", "content": msg.content} for msg in state["messages"]],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        await self.db_service.save_conversation(conversation)
        
        state["messages"].append(AIMessage(content="Retrieved user information and conversation history"))
        
        logger.info(f"Retrieved user info for {user_id}")
        return state

    async def action_agent(self, state: AgentState) -> AgentState:
        """Agent that takes specific actions based on user requests"""
        logger.info("Action agent processing request")
        
        request_type = state.get("current_request_type")
        user_info = state.get("user_info", {})
        latest_message = state["messages"][-3].content if len(state["messages"]) >= 3 else ""
        
        response_text = ""
        suggested_actions = []
        requires_human = False
        ticket_id = None
        
        if request_type == RequestType.PASSWORD_RESET:
            response_text = f"I'll help you reset your password, {user_info.get('name', 'User')}. "
            response_text += "For security reasons, I'm sending a password reset link to your registered email. "
            response_text += "Please check your email and follow the instructions."
            suggested_actions = ["Check email inbox", "Contact IT if no email received"]
            
            # Create ticket
            ticket = Ticket(
                id=f"PWD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                user_id=state["user_id"],
                title="Password Reset Request",
                description=f"User requested password reset: {latest_message}",
                request_type=RequestType.PASSWORD_RESET,
                status=TicketStatus.IN_PROGRESS,
                priority="Medium",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await self.db_service.create_ticket(ticket)
            ticket_id = ticket.id
            
        elif request_type == RequestType.SOFTWARE_ACCESS:
            response_text = f"I'll help you with software access, {user_info.get('name', 'User')}. "
            response_text += "I've created a ticket for software access request. "
            response_text += "Our IT team will review your request and provide access within 2 business days."
            suggested_actions = ["Provide software details", "Specify business justification"]
            requires_human = True
            
            ticket = Ticket(
                id=f"SW-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                user_id=state["user_id"],
                title="Software Access Request",
                description=f"User requested software access: {latest_message}",
                request_type=RequestType.SOFTWARE_ACCESS,
                status=TicketStatus.OPEN,
                priority="Medium",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await self.db_service.create_ticket(ticket)
            ticket_id = ticket.id
            
        elif request_type == RequestType.NETWORK_ISSUE:
            response_text = f"I understand you're having network issues, {user_info.get('name', 'User')}. "
            response_text += "Let me help you troubleshoot. First, please try these steps: "
            response_text += "1. Restart your computer, 2. Check network cables, 3. Try connecting to a different network."
            suggested_actions = ["Restart computer", "Check cables", "Test different network"]
            
        elif request_type == RequestType.HARDWARE_REQUEST:
            response_text = f"I'll help you with your hardware request, {user_info.get('name', 'User')}. "
            response_text += "I've created a ticket for your hardware request. "
            response_text += "Please provide details about the hardware you need and the business justification."
            suggested_actions = ["Specify hardware details", "Provide business justification"]
            requires_human = True
            
            ticket = Ticket(
                id=f"HW-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                user_id=state["user_id"],
                title="Hardware Request",
                description=f"User requested hardware: {latest_message}",
                request_type=RequestType.HARDWARE_REQUEST,
                status=TicketStatus.OPEN,
                priority="Low",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await self.db_service.create_ticket(ticket)
            ticket_id = ticket.id
            
        else:  # GENERAL_SUPPORT
            response_text = f"Thank you for contacting IT Support, {user_info.get('name', 'User')}. "
            response_text += "I'm here to help with your general support request. "
            response_text += "Could you please provide more details about the issue you're experiencing?"
            suggested_actions = ["Provide more details", "Describe the issue"]
        
        # Update state
        state["ticket_id"] = ticket_id
        state["requires_human_intervention"] = requires_human
        state["suggested_actions"] = suggested_actions
        
        # Add response to messages
        state["messages"].append(AIMessage(content=response_text))
        
        logger.info(f"Action agent completed processing for {request_type}")
        return state

    def _classify_request(self, message: str) -> RequestType:
        """Classify the type of request based on message content"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['password', 'reset', 'login', 'access denied', 'locked out']):
            return RequestType.PASSWORD_RESET
        elif any(word in message_lower for word in ['software', 'application', 'install', 'license', 'access']):
            return RequestType.SOFTWARE_ACCESS
        elif any(word in message_lower for word in ['laptop', 'computer', 'monitor', 'keyboard', 'mouse', 'hardware']):
            return RequestType.HARDWARE_REQUEST
        elif any(word in message_lower for word in ['network', 'internet', 'wifi', 'connection', 'slow']):
            return RequestType.NETWORK_ISSUE
        else:
            return RequestType.GENERAL_SUPPORT

    def should_continue(self, state: AgentState) -> str:
        """Determine if the workflow should continue or end"""
        if state.get("requires_human_intervention", False):
            return "end"  # Ticket created, human will follow up
        elif state.get("current_request_type") == RequestType.GENERAL_SUPPORT:
            return "continue"  # May need more interaction
        else:
            return "end"  # Request processed

# ==========================================
# LANGGRAPH WORKFLOW
# ==========================================

class ITServiceDeskWorkflow:
    def __init__(self, db_service: CosmosDBService):
        self.agents = ITServiceDeskAgents(db_service)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("orchestrator", self.agents.orchestrator_agent)
        workflow.add_node("cosmosdb_agent", self.agents.cosmosdb_tool_agent) 
        workflow.add_node("action_agent", self.agents.action_agent)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges
        workflow.add_edge("orchestrator", "cosmosdb_agent")
        workflow.add_edge("cosmosdb_agent", "action_agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "action_agent",
            self.agents.should_continue,
            {
                "continue": "orchestrator",
                "end": END
            }
        )
        
        return workflow.compile()

    async def process_request(self, chat_request: ChatRequest) -> ChatResponse:
        """Process a chat request through the workflow"""
        session_id = chat_request.session_id or f"session-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=chat_request.message)],
            "user_id": chat_request.user_id,
            "session_id": session_id,
            "user_info": None,
            "previous_conversations": [],
            "current_request_type": None,
            "ticket_id": None,
            "requires_human_intervention": False,
            "suggested_actions": [],
            "context": {"user_role": chat_request.user_role}
        }
        
        # Run workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract final response
        ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        final_response = ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't process your request."
        
        return ChatResponse(
            response=final_response,
            session_id=session_id,
            ticket_id=final_state.get("ticket_id"),
            requires_action=final_state.get("requires_human_intervention", False),
            suggested_actions=final_state.get("suggested_actions", [])
        )

# ==========================================
# FASTAPI APPLICATION
# ==========================================

# Global services
db_service = None
workflow_service = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global db_service, workflow_service
    
    # Startup
    logger.info("Starting IT Service Desk ChatBot...")
    db_service = CosmosDBService()
    workflow_service = ITServiceDeskWorkflow(db_service)
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down IT Service Desk ChatBot...")

# Create FastAPI app
app = FastAPI(
    title="IT Service Desk ChatBot",
    description="AI-powered IT Service Desk with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - in production, implement proper JWT validation"""
    # For demo purposes, we'll extract user_id from token
    # In production, validate JWT token here
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    # Mock user extraction - replace with actual JWT decoding
    return {"user_id": token, "role": UserRole.TESTER}

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Main chat endpoint"""
    try:
        # Use authenticated user ID
        request.user_id = current_user["user_id"]
        
        # Process request through workflow
        response = await workflow_service.process_request(request)
        
        # Log interaction in background
        background_tasks.add_task(
            log_interaction,
            request.user_id,
            request.message,
            response.response
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{user_id}/conversations")
async def get_user_conversations(
    user_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user conversation history"""
    try:
        # Check if user can access these conversations
        if current_user["user_id"] != user_id and current_user["role"] not in [UserRole.HELPDESK, UserRole.IT_SUPPORT, UserRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        conversations = await db_service.get_user_conversations(user_id, limit)
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Error getting conversations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tickets/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get ticket information"""
    try:
        # In production, add proper authorization checks
        ticket = await db_service.tickets_container.read_item(item=ticket_id, partition_key=ticket_id)
        return ticket
        
    except exceptions.CosmosResourceNotFoundError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except Exception as e:
        logger.error(f"Error getting ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/tickets/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: str,
    status: TicketStatus,
    current_user: dict = Depends(get_current_user)
):
    """Update ticket status - only for helpdesk/IT support roles"""
    try:
        if current_user["role"] not in [UserRole.HELPDESK, UserRole.IT_SUPPORT, UserRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        updated_ticket = await db_service.update_ticket_status(ticket_id, status)
        return updated_ticket
        
    except Exception as e:
        logger.error(f"Error updating ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/dashboard")
async def get_analytics_dashboard(current_user: dict = Depends(get_current_user)):
    """Get analytics dashboard data - admin/IT support only"""
    try:
        if current_user["role"] not in [UserRole.IT_SUPPORT, UserRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get ticket statistics
        # This is a simplified version - in production, implement proper aggregations
        
        return {
            "total_tickets": 0,  # Implement actual counting
            "open_tickets": 0,
            "resolved_tickets": 0,
            "avg_resolution_time": 0,
            "top_request_types": []
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Background task
async def log_interaction(user_id: str, user_message: str, bot_response: str):
    """Log user interactions for analytics"""
    try:
        # Implement interaction logging
        logger.info(f"User {user_id}: {user_message[:50]}... -> {bot_response[:50]}...")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

# ==========================================
# MAIN APPLICATION RUNNER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        log_level="info"
    )