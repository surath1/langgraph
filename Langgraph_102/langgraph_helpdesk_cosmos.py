"""
IT Service Desk ChatBot using LangGraph
A multi-agent system for handling IT support requests with CosmosDB integration
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Azure Cosmos DB imports
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "your-cosmos-endpoint")
COSMOS_KEY = os.getenv("COSMOS_KEY", "your-cosmos-key")
DATABASE_NAME = "ITServiceDesk"
USER_CONTAINER = "Users"
TICKETS_CONTAINER = "Tickets"

# Initialize Cosmos DB client
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
user_container = database.create_container_if_not_exists(
    id=USER_CONTAINER,
    partition_key=PartitionKey(path="/user_id"),
    offer_throughput=400
)
tickets_container = database.create_container_if_not_exists(
    id=TICKETS_CONTAINER,
    partition_key=PartitionKey(path="/user_id"),
    offer_throughput=400
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    user_id: str
    user_details: Optional[Dict]
    ticket_id: Optional[str]
    request_type: Optional[str]
    request_details: Optional[Dict]
    action_required: Optional[str]
    next_agent: Optional[str]
    resolved: bool

@dataclass
class UserDetails:
    user_id: str
    name: str
    email: str
    department: str
    role: str
    manager: str
    last_login: str
    active_tickets: List[str]
    previous_requests: List[Dict]

@dataclass
class Ticket:
    ticket_id: str
    user_id: str
    request_type: str
    description: str
    priority: str
    status: str
    created_at: str
    updated_at: str
    assigned_to: Optional[str]
    resolution: Optional[str]

# Tools for CosmosDB operations
@tool
def get_user_details(user_id: str) -> Dict:
    """Retrieve user details from CosmosDB"""
    try:
        user_doc = user_container.read_item(item=user_id, partition_key=user_id)
        return user_doc
    except CosmosResourceNotFoundError:
        return {"error": "User not found"}

@tool
def create_user_entry(user_data: Dict) -> Dict:
    """Create a new user entry in CosmosDB"""
    try:
        user_data["id"] = user_data["user_id"]
        user_data["created_at"] = datetime.now().isoformat()
        user_container.create_item(body=user_data)
        return {"success": True, "message": "User created successfully"}
    except CosmosResourceExistsError:
        return {"error": "User already exists"}

@tool
def get_user_previous_requests(user_id: str) -> List[Dict]:
    """Get user's previous support requests"""
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.created_at DESC"
        items = list(tickets_container.query_items(query=query, enable_cross_partition_query=True))
        return items[:10]  # Return last 10 requests
    except Exception as e:
        return [{"error": str(e)}]

@tool
def create_ticket(ticket_data: Dict) -> Dict:
    """Create a new support ticket"""
    try:
        ticket_id = f"TICK-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ticket_data["id"] = ticket_id
        ticket_data["ticket_id"] = ticket_id
        ticket_data["created_at"] = datetime.now().isoformat()
        ticket_data["updated_at"] = datetime.now().isoformat()
        ticket_data["status"] = "Open"
        
        tickets_container.create_item(body=ticket_data)
        return {"success": True, "ticket_id": ticket_id}
    except Exception as e:
        return {"error": str(e)}

@tool
def update_ticket_status(ticket_id: str, status: str, resolution: Optional[str] = None) -> Dict:
    """Update ticket status and resolution"""
    try:
        # Get the ticket first
        query = f"SELECT * FROM c WHERE c.ticket_id = '{ticket_id}'"
        tickets = list(tickets_container.query_items(query=query, enable_cross_partition_query=True))
        
        if tickets:
            ticket = tickets[0]
            ticket["status"] = status
            ticket["updated_at"] = datetime.now().isoformat()
            if resolution:
                ticket["resolution"] = resolution
            
            tickets_container.replace_item(item=ticket["id"], body=ticket)
            return {"success": True, "message": "Ticket updated successfully"}
        else:
            return {"error": "Ticket not found"}
    except Exception as e:
        return {"error": str(e)}

# Create tool executor
tools = [get_user_details, create_user_entry, get_user_previous_requests, create_ticket, update_ticket_status]
tool_executor = ToolExecutor(tools)

# Agent 1: Orchestrator Agent
def orchestrator_agent(state: AgentState) -> AgentState:
    """
    Main orchestrator that routes requests and coordinates between agents
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Initial request processing
    if not state.get("user_id"):
        # Extract user ID from the conversation or ask for it
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an IT Service Desk Orchestrator. Your job is to:
            1. Identify the user and their request type
            2. Route requests to appropriate agents
            3. Coordinate responses between agents
            
            Common request types:
            - SOFTWARE_ACCESS: Software installation, license requests
            - PASSWORD_RESET: Password reset requests
            - HARDWARE_ISSUE: Hardware problems, equipment requests
            - NETWORK_ISSUE: Network connectivity problems
            - ACCOUNT_ISSUE: Account creation, permissions, access issues
            
            First, identify the user ID and request type from the user's message.
            If user ID is not provided, ask for it politely."""),
            ("human", "{input}")
        ])
        
        response = llm.invoke(prompt.format_messages(input=last_message.content if last_message else ""))
        
        # Simple user ID extraction (in real implementation, use better NLP)
        user_input = last_message.content if last_message else ""
        
        # For demo purposes, assume user provides ID or we can extract it
        state["user_id"] = "user123"  # This should be extracted from the conversation
        state["next_agent"] = "database_agent"
        
        return state
    
    # Route based on current state
    if state.get("user_details") and not state.get("action_required"):
        # User details retrieved, now process the request
        state["next_agent"] = "action_agent"
    elif state.get("action_required"):
        # Action completed, check if resolved
        if state.get("resolved"):
            response = AIMessage(content="Your request has been processed successfully. Is there anything else I can help you with?")
            state["messages"].append(response)
            state["next_agent"] = None
        else:
            # More actions needed
            state["next_agent"] = "action_agent"
    else:
        # Need user details first
        state["next_agent"] = "database_agent"
    
    return state

# Agent 2: Database Agent
def database_agent(state: AgentState) -> AgentState:
    """
    Handles all CosmosDB operations including user lookup and ticket management
    """
    user_id = state["user_id"]
    
    # Get user details
    user_details = get_user_details.invoke({"user_id": user_id})
    
    if "error" in user_details:
        # User not found, create basic entry
        basic_user_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"{user_id}@company.com",
            "department": "Unknown",
            "role": "Employee",
            "manager": "Unknown",
            "last_login": datetime.now().isoformat(),
            "active_tickets": [],
            "previous_requests": []
        }
        create_result = create_user_entry.invoke({"user_data": basic_user_data})
        user_details = basic_user_data
    
    # Get previous requests
    previous_requests = get_user_previous_requests.invoke({"user_id": user_id})
    user_details["previous_requests"] = previous_requests
    
    state["user_details"] = user_details
    state["next_agent"] = "orchestrator"
    
    # Add context message
    context_msg = f"Found user: {user_details.get('name', 'Unknown')} from {user_details.get('department', 'Unknown')} department."
    if previous_requests and not any("error" in req for req in previous_requests):
        context_msg += f" User has {len(previous_requests)} previous requests."
    
    state["messages"].append(AIMessage(content=context_msg))
    
    return state

# Agent 3: Action Agent
def action_agent(state: AgentState) -> AgentState:
    """
    Processes user requests and takes appropriate actions
    """
    messages = state["messages"]
    user_details = state["user_details"]
    last_user_message = None
    
    # Find the last human message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break
    
    if not last_user_message:
        state["messages"].append(AIMessage(content="I need more information about your request."))
        return state
    
    # Analyze the request
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an IT Support Action Agent. Analyze the user's request and determine:
        1. Request type (SOFTWARE_ACCESS, PASSWORD_RESET, HARDWARE_ISSUE, NETWORK_ISSUE, ACCOUNT_ISSUE)
        2. Priority (High, Medium, Low)
        3. Required actions
        4. Whether this can be resolved immediately or needs escalation
        
        User Details: {user_details}
        Previous Requests: {previous_requests}
        
        Provide your analysis in JSON format:
        {{
            "request_type": "type",
            "priority": "priority_level",
            "description": "detailed_description",
            "actions": ["action1", "action2"],
            "can_resolve": true/false,
            "resolution": "resolution_if_applicable"
        }}"""),
        ("human", "{request}")
    ])
    
    analysis_response = llm.invoke(prompt.format_messages(
        user_details=str(user_details),
        previous_requests=str(user_details.get("previous_requests", [])),
        request=last_user_message.content
    ))
    
    try:
        # Parse the analysis (in real implementation, use structured output)
        analysis = {
            "request_type": "SOFTWARE_ACCESS",  # Default for demo
            "priority": "Medium",
            "description": last_user_message.content,
            "actions": ["Create ticket", "Assign to IT team"],
            "can_resolve": False,
            "resolution": None
        }
        
        # Create ticket
        ticket_data = {
            "user_id": state["user_id"],
            "request_type": analysis["request_type"],
            "description": analysis["description"],
            "priority": analysis["priority"],
            "assigned_to": "IT Support Team"
        }
        
        ticket_result = create_ticket.invoke({"ticket_data": ticket_data})
        
        if ticket_result.get("success"):
            ticket_id = ticket_result["ticket_id"]
            state["ticket_id"] = ticket_id
            
            if analysis["can_resolve"]:
                # Mark as resolved
                update_ticket_status.invoke({
                    "ticket_id": ticket_id,
                    "status": "Resolved",
                    "resolution": analysis.get("resolution", "Request processed successfully")
                })
                response_msg = f"Your request has been processed successfully. Ticket #{ticket_id} has been resolved."
                state["resolved"] = True
            else:
                response_msg = f"I've created ticket #{ticket_id} for your request. Our IT team will process this within 24 hours."
                state["resolved"] = False
            
            state["messages"].append(AIMessage(content=response_msg))
            state["action_required"] = "completed"
        else:
            state["messages"].append(AIMessage(content="I'm sorry, there was an issue creating your ticket. Please try again."))
            state["resolved"] = False
    
    except Exception as e:
        state["messages"].append(AIMessage(content=f"An error occurred while processing your request: {str(e)}"))
        state["resolved"] = False
    
    state["next_agent"] = "orchestrator"
    return state

# Router function
def route_to_next_agent(state: AgentState) -> str:
    """Route to the next agent based on the current state"""
    next_agent = state.get("next_agent")
    
    if next_agent == "database_agent":
        return "database_agent"
    elif next_agent == "action_agent":
        return "action_agent"
    elif next_agent == "orchestrator":
        return "orchestrator_agent"
    else:
        return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator_agent", orchestrator_agent)
workflow.add_node("database_agent", database_agent)
workflow.add_node("action_agent", action_agent)

# Add edges
workflow.set_entry_point("orchestrator_agent")
workflow.add_conditional_edges(
    "orchestrator_agent",
    route_to_next_agent,
    {
        "database_agent": "database_agent",
        "action_agent": "action_agent",
        END: END
    }
)
workflow.add_conditional_edges(
    "database_agent",
    route_to_next_agent,
    {
        "orchestrator_agent": "orchestrator_agent",
        END: END
    }
)
workflow.add_conditional_edges(
    "action_agent",
    route_to_next_agent,
    {
        "orchestrator_agent": "orchestrator_agent",
        END: END
    }
)

# Compile the graph
app = workflow.compile()

# Chat interface
class ITServiceDeskChatBot:
    def __init__(self):
        self.app = app
        
    def process_request(self, user_message: str, user_id: str = None) -> str:
        """Process a user request through the multi-agent system"""
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id or "user123",  # Default for demo
            "user_details": None,
            "ticket_id": None,
            "request_type": None,
            "request_details": None,
            "action_required": None,
            "next_agent": None,
            "resolved": False
        }
        
        # Run the workflow
        final_state = self.app.invoke(initial_state)
        
        # Extract the final response
        ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content
        else:
            return "I'm sorry, I couldn't process your request. Please try again."
    
    def chat(self):
        """Interactive chat interface"""
        print("🤖 IT Service Desk ChatBot")
        print("=" * 50)
        print("Hi! I'm your IT Support assistant. I can help with:")
        print("- Software access requests")
        print("- Password resets")
        print("- Hardware issues")
        print("- Network problems")
        print("- Account issues")
        print("\nType 'quit' to exit")
        print("=" * 50)
        
        while True:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Goodbye! Have a great day!")
                break
            
            try:
                response = self.process_request(user_input)
                print(f"🤖 IT Support: {response}")
            except Exception as e:
                print(f"🤖 IT Support: I'm sorry, I encountered an error: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = ITServiceDeskChatBot()
    
    # Example requests for testing
    test_requests = [
        "I need access to Microsoft Office 365",
        "I forgot my password and can't log in",
        "My laptop screen is flickering",
        "I can't connect to the company WiFi",
        "I need a new user account created for a new employee"
    ]
    
    print("Testing the IT Service Desk ChatBot:")
    print("=" * 50)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\nTest {i}: {request}")
        try:
            response = chatbot.process_request(request, f"test_user_{i}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Starting interactive chat...")
    
    # Start interactive chat
    # chatbot.chat()