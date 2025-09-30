import json
import os
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging
from dotenv import load_dotenv
load_dotenv()  

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("No Gemini credentials found: set GOOGLE_API_KEY run gcloud auth application-default login")

# Define tools
@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for relevant information."""
    knowledge = {
        "refund": "Refunds can be processed within 30 days. Please provide order number.",
        "shipping": "Standard shipping takes 3-5 business days. Express takes 1-2 days.",
        "returns": "Returns are accepted within 30 days in original condition.",
        "account": "Account issues can be resolved by resetting password or contacting support.",
        "payment": "We accept credit cards, PayPal, and bank transfers.",
        "warranty": "Products have 1-year warranty covering manufacturing defects."
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return f"Knowledge Base Result: {value}"
    return "No relevant information found in knowledge base."


@tool
def run_diagnostics(system_name: str) -> str:
    """Run a simulated diagnostics check for a system and return results."""
    # In a real implementation this would call monitoring APIs or run checks
    checks = {
        "email": "Email service: reachable, queue length normal, no recent failures.",
        "vpn": "VPN status: degraded for 2% of users. Restart recommended.",
        "printer": "Printer: offline. Paper jam detected.",
        "database": "Database: high latency observed on primary node. Investigating."
    }
    return checks.get(system_name.lower(), f"No diagnostics available for {system_name}.")


@tool
def reset_password(user_identifier: str) -> str:
    """Simulate a password reset process for a user identifier (email or username)."""
    reset_token = uuid.uuid4().hex[:6].upper()
    return f"Password reset initiated for {user_identifier}. Temporary code: {reset_token}. Invalidate after 15 minutes."


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Lookup an existing support ticket in a mocked ticket system."""
    tickets = {
        "TKT1001": "TKT1001: Password reset completed. Closed.",
        "TKT1002": "TKT1002: VPN connectivity issue. In progress.",
        "TKT1003": "TKT1003: Printer replacement scheduled.",
    }
    return tickets.get(ticket_id.upper(), f"Ticket {ticket_id} not found.")

@tool
def check_order_status(order_id: str) -> str:
    """Check the status of a customer order by order ID. if is exist's don't assume"""
    orders = {
        "ORD001": "Order shipped on 2024-01-15. Tracking: TRK123456. Expected delivery: 2024-01-18.",
        "ORD002": "Order processing. Payment confirmed. Will ship within 24 hours.",
        "ORD003": "Order delivered on 2024-01-10. Customer signature received.",
        "ORD004": "Order cancelled at customer request. Refund processed."
    }
    if order_id.upper() in orders:
        return f"Order Status: {orders[order_id.upper()]}"
    return f"Order {order_id} not found. Please verify the order number."

@tool
def escalate_to_human(reason: str, customer_info: str) -> str:
    """Escalate the conversation to a human agent with a reason and customer details."""
    escalation_id = f"ESC_{uuid.uuid4().hex[:8].upper()}"
    return f"Escalated to human agent. Ticket ID: {escalation_id}. Reason: {reason}. Customer: {customer_info}"

@tool
def create_support_ticket(issue: str, priority: Literal["low", "medium", "high"] = "medium") -> str:
    """Create a support ticket for a customer issue with specified priority."""
    ticket_id = f"TKT_{uuid.uuid4().hex[:8].upper()}"
    return f"Support ticket created: {ticket_id}. Issue: {issue}. Priority: {priority}. Expected response: 24-48 hours."

# Define state schema
class CustomerSupportState(MessagesState):
    """State schema for the customer support workflow.

    Notes:
    - `messages` is provided by MessagesState from langgraph.
    - `tools_used` is an accumulating list of tool names called by the LLM.
    """
    customer_query: str
    tools_used: Annotated[List[str], add]
    escalation_needed: bool = False
    issue_resolved: bool = False
    confidence_score: float = 0.0
    current_step: str = ""
    # IT-helpdesk specific fields
    issue_type: Optional[str] = None
    priority: Optional[str] = None

def serialize_message(message: BaseMessage) -> Dict:
    """Convert a BaseMessage to a JSON-serializable dictionary."""
    serialized = {
        "type": type(message).__name__,
        "content": message.content,
        "id": getattr(message, 'id', None),
        "additional_kwargs": getattr(message, 'additional_kwargs', {})
    }
    if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
        serialized["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args"), "id": call.get("id")}
            for call in message.tool_calls
        ]
    elif isinstance(message, ToolMessage):
        serialized["tool_call_id"] = getattr(message, 'tool_call_id', None)
    return serialized

def serialize_state(state: Dict) -> Dict:
    """Convert state to a JSON-serializable dictionary."""
    serialized_state = {
        "messages": [],
        "customer_query": state.get("customer_query", ""),
        "tools_used": state.get("tools_used", []),
        "escalation_needed": state.get("escalation_needed", False),
        "issue_resolved": state.get("issue_resolved", False),
        "confidence_score": state.get("confidence_score", 0.0),
        "current_step": state.get("current_step", ""),
        "issue_type": state.get("issue_type"),
        "priority": state.get("priority")
    }
    # Collect messages from different nested parts of the state.
    messages: List[BaseMessage] = []
    # If state directly contains messages
    if isinstance(state.get("messages"), list):
        messages.extend(state.get("messages"))
    # Inspect nested dicts that may contain messages
    for key, value in state.items():
        if isinstance(value, dict) and "messages" in value and isinstance(value["messages"], list):
            messages.extend(value["messages"])
    # Some values may already be Message objects in other structures
    serialized_state["messages"] = [serialize_message(msg) for msg in messages if isinstance(msg, BaseMessage)]
    return serialized_state

# Create LLM with tools
def create_llm_with_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Latest model
        temperature=0.1,  # Lower temperature for more consistent responses
        max_retries=2,
        google_api_key=GOOGLE_API_KEY,
    )
    
    tools = [search_knowledge_base, check_order_status, escalate_to_human, create_support_ticket]
    return llm.bind_tools(tools), tools

# Agent node
async def customer_service_agent(state: CustomerSupportState):
    logger.debug("Entering customer_service_agent")
    messages = state["messages"]
    logger.info(f"customer_service_agent messages: {messages}")
    llm_with_tools, _ = create_llm_with_tools()
    system_message = SystemMessage(content="""You are an IT Helpdesk assistant. You can:
        1. Search the knowledge base for information
        2. Check order status using order IDs
        3. Create support tickets for complex issues
        4. Escalate to human agents when needed
        Be professional, triage issues, and use diagnostic tools when appropriate.
        If you think no human input is required for such message, you MUST start your response with "FINAL_MESSAGE_NO_HUMAN_IN_LOOP_REQUIRED": followed by a <AI MESSAGE content>
        When requesting human input (e.g., assist,order number, account details), you MUST start your response with 'HUMAN_IN_LOOP:' followed by a context ('Assist Related','Order related', 'Account issue', 'Payment issue', 'General query'), followed by a colon and the message. The format must be:
        HUMAN_IN_LOOP: <context>: <message>
        Examples:
        - HUMAN_IN_LOOP: Assist: I am an AI customer service agent. I'm here to assist you with various tasks such as searching for information, checking the status of your orders, creating support tickets for complex issues, and escalating matters to human agents when necessary. How may I assist you today?
        - HUMAN_IN_LOOP: Order related: Our refund policy allows for refunds to be processed within 30 days of purchase. Please provide the order number.
        - HUMAN_IN_LOOP: Account issue: Please provide your account email to resolve this issue.
        Do NOT append or place the prefix anywhere else in the message. If you cannot resolve an issue, use 'HUMAN_IN_LOOP: Escalation needed: [reason]' to escalate to a human agent.
        """)
    # Call the LLM (with tool bindings). The LLM may perform tool calls, which will be returned
    # attached to the AIMessage as `.tool_calls` depending on the integrations.
    response = await llm_with_tools.ainvoke([system_message] + messages)

    # Update state: append the AI response and record any tools used
    tools_called: List[str] = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            name = call.get("name") or call.get("tool_name") or None
            if name:
                tools_called.append(name)

    updated_state = {
        **state,
        "messages": state.get("messages", []) + [response],
        "tools_used": list(dict.fromkeys(state.get("tools_used", []) + tools_called)),
        "current_step": "agent"
    }
    logger.debug("Exiting customer_service_agent")
    return updated_state


# Triage node: determines issue_type and priority before agent handles
async def triage_node(state: CustomerSupportState):
    logger.debug("Entering triage_node")
    # Very small heuristic triage based on keywords
    content = ""
    msgs = state.get("messages", [])
    if msgs:
        first = msgs[0]
        content = getattr(first, 'content', '')

    issue_type = None
    priority = "low"
    if any(k in content.lower() for k in ["password", "reset"]):
        issue_type = "password"
        priority = "high"
    elif any(k in content.lower() for k in ["vpn", "connect", "latency", "network"]):
        issue_type = "network"
        priority = "medium"
    elif any(k in content.lower() for k in ["printer", "paper", "jam"]):
        issue_type = "hardware"
        priority = "low"
    else:
        issue_type = "general"

    updated_state = {
        **state,
        "issue_type": issue_type,
        "priority": priority,
        "current_step": "triage"
    }
    logger.info(f"triage_node determined issue_type={issue_type}, priority={priority}")
    logger.debug("Exiting triage_node")
    return updated_state

# Tool node
def create_tool_node():
    _, tools = create_llm_with_tools()
    # Use ToolNode to expose the tools to the graph. ToolNode will execute and return ToolMessages
    # which the agent node can inspect via AIMessage.tool_calls.
    return ToolNode(tools)

# Human review node
async def human_review_node(state: CustomerSupportState):
    logger.debug("Entering human_review_node")
    # Human node is an interrupt point. The conversation will pause and wait for external input.
    updated_state = {
        **state,
        "current_step": "human_review",
    }
    logger.info(f"human_review_node state: {updated_state}")
    logger.debug("Exiting human_review_node")
    return updated_state

# Resolution check node
async def resolution_check_node(state: CustomerSupportState):
    logger.debug("Entering resolution_check_node")
    messages = state["messages"]
    tools_used = state.get("tools_used", [])
    confidence = 0.8 if len(tools_used) > 0 and len(messages) >= 2 else 0.4
    resolved = confidence > 0.7
    updated_state = {
        **state,
        "confidence_score": confidence,
        "issue_resolved": resolved,
        "current_step": "resolution_check"
    }
    logger.info(f"resolution_check_node: confidence={confidence}, resolved={resolved}")
    logger.debug("Exiting resolution_check_node")
    return updated_state

# Conditional routing
def should_continue_or_escalate(state: CustomerSupportState) -> str:
    logger.debug("Entering should_continue_or_escalate")
    messages = state["messages"]
    if not messages:
        decision = "agent"
    else:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            decision = "tools"
        elif isinstance(last_message, AIMessage):
            content = last_message.content
            # Look for clear prefixes defined in system prompt
            if "FINAL_MESSAGE_NO_HUMAN_IN_LOOP_REQUIRED" in content:
                decision = "end"
            elif "HUMAN_IN_LOOP:" in content:
                # If the model requests human input, go to human review
                decision = "human_review"
            else:
                decision = "check_resolution"
        else:
            decision = "agent"
    logger.info(f"should_continue_or_escalate decision: {decision}")
    logger.debug("Exiting should_continue_or_escalate")
    return decision

def should_end_or_continue(state: CustomerSupportState) -> str:
    logger.debug("Entering should_end_or_continue")
    confidence = state.get("confidence_score", 0.0)
    resolved = state.get("issue_resolved", False)
    if resolved and confidence > 0.7:
        decision = "end"
    elif confidence < 0.3:
        decision = "human_review"
    else:
        decision = "agent"
    logger.info(f"should_end_or_continue decision: {decision}")
    logger.debug("Exiting should_end_or_continue")
    return decision

# Create workflow graph
def create_customer_support_graph():
    workflow = StateGraph(CustomerSupportState)
    # add triage node before agent
    workflow.add_node("triage", triage_node)
    workflow.add_node("agent", customer_service_agent)
    workflow.add_node("tools", create_tool_node())
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("check_resolution", resolution_check_node)
    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue_or_escalate,
        {"end": END,"tools": "tools", "human_review": "human_review", "check_resolution": "check_resolution"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("human_review", "agent")
    workflow.add_conditional_edges(
        "check_resolution",
        should_end_or_continue,
        {"end": END, "human_review": "human_review", "agent": "agent"}
    )
    # Use MemorySaver by default, but allow production replacement via env var later
    checkpointer = MemorySaver()
    # Interrupt before human_review so external systems can inject human feedback
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"]) 

# Initialize FastAPI app
app = FastAPI(title="Customer Support Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create global graph instance
support_graph = create_customer_support_graph()

# Request models
class SupportRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class HumanFeedback(BaseModel):
    feedback: str
    action: Literal["continue", "escalate", "resolve"] = "continue"

# API Endpoints
@app.post("/api/chat")
async def handle_support_chat(request: SupportRequest):
    """Handle customer support queries with streaming."""
    logger.debug("Entering handle_support_chat")
    logger.info(f"Received support chat request: {request}")
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # system_message = SystemMessage(content="""You are a helpful customer service agent. You can:
    # 1. Search the knowledge base for information
    # 2. Check order status using order IDs
    # 3. Create support tickets for complex issues
    # 4. Escalate to human agents when needed
    # Be friendly, helpful, and use tools when appropriate.
    # When requesting human input (e.g., order number, account details), you MUST start your response with 'HUMAN_IN_LOOP:' followed by a context ('Order related', 'Account issue', 'Payment issue', 'General query'), followed by a colon and the message. The format must be:
    # HUMAN_IN_LOOP: <context>: <message>
    # Examples:
    # - HUMAN_IN_LOOP: Order related: Our refund policy allows for refunds to be processed within 30 days of purchase. Please provide the order number.
    # - HUMAN_IN_LOOP: Account issue: Please provide your account email to resolve this issue.
    # Do NOT append or place the prefix anywhere else in the message. If you cannot resolve an issue, use 'HUMAN_IN_LOOP: Escalation needed: [reason]' to escalate to a human agent.
    # """)
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "customer_query": request.query,
        "tools_used": [],
        "escalation_needed": False,
        "issue_resolved": False,
        "confidence_score": 0.0,
        "current_step": "initial"
    }
    logger.info(f"Initial state for thread {thread_id}: {initial_state}")
    async def event_stream():
        try:
            async for chunk in support_graph.astream(input=initial_state, config=config):
                # Each chunk is a state snapshot from the graph. Provide structured SSE events.
                logger.debug(f"Graph chunk: {chunk}")
                serialized_chunk = serialize_state(chunk)
                event = {
                    'type': 'graph_chunk',
                    'thread_id': thread_id,
                    'chunk': serialized_chunk,
                    'timestamp': datetime.now().isoformat()
                }
                # Add meta information when available
                if hasattr(chunk, 'tasks'):
                    event['tasks'] = getattr(chunk, 'tasks')
                if hasattr(chunk, 'interrupts'):
                    event['interrupts'] = getattr(chunk, 'interrupts')
                logger.debug(f"Serialized chunk: {event}")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in event_stream: {error_trace}")
            yield f"data: {json.dumps({'type': 'error', 'error': error_trace})}\n\n"
    logger.debug("Exiting handle_support_chat")
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# (legacy human_response endpoint removed during cleanup)

@app.get("/api/status/{thread_id}")
async def get_support_status(thread_id: str):
    """Retrieve the current status of a support conversation."""
    logger.debug(f"Entering get_support_status for thread {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = support_graph.get_state(config)
        snapshot_dict = {
            'values': snapshot.values,
            'next': snapshot.next,
            'config': snapshot.config,
            'metadata': snapshot.metadata,
            'created_at': snapshot.created_at,
            'parent_config': snapshot.parent_config,
            'tasks': snapshot.tasks,
            'interrupts': snapshot.interrupts
        }
        logger.info(f"Support status snapshot: {snapshot_dict}")
        logger.debug("Exiting get_support_status")
        return snapshot_dict
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in get_support_status: {error_trace}")
        raise HTTPException(status_code=500, detail=error_trace)


@app.post("/api/feedback/{thread_id}")
async def provide_human_feedback(thread_id: str, feedback: HumanFeedback):
    """Process human feedback for an interrupted conversation."""
    logger.info(f"Processing feedback for thread_id: {thread_id}, feedback: {feedback.feedback}")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        current_state = support_graph.get_state(config)
        if not current_state:
            logger.warning(f"Thread not found: {thread_id}")
            raise HTTPException(status_code=404, detail="Thread not found")

        # Build human message and update state via the interrupt node
        human_message = HumanMessage(content=f"Human feedback: {feedback.feedback}")
        # Update flags based on action
        if feedback.action == "escalate":
            # Mark escalation needed and send human message
            support_graph.update_state(config, {"messages": human_message, "escalation_needed": True}, as_node="human_review")
        elif feedback.action == "resolve":
            support_graph.update_state(config, {"messages": human_message, "issue_resolved": True, "confidence_score": 1.0}, as_node="human_review")
        else:
            support_graph.update_state(config, {"messages": human_message}, as_node="human_review")

        # Resume the graph and gather resulting chunks until completion or next interrupt
        result = []
        async for chunk in support_graph.astream(None, config=config):
            result.append(serialize_state(chunk))

        logger.info(f"Feedback processed successfully for thread_id: {thread_id}")
        return {
            "status": "feedback_processed",
            "thread_id": thread_id,
            "action": feedback.action,
            "result_chunks": len(result),
            "final_chunks": result
        }
    except Exception as e:
        logger.error(f"Error processing feedback for thread_id: {thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/api/conversation/{thread_id}")
async def get_conversation(thread_id: str):
    """Return the current conversation messages for a thread in serialized form."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = support_graph.get_state(config)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Thread not found")
        state_values = snapshot.values
        return serialize_state(state_values)
    except Exception as e:
        logger.error(f"Error in get_conversation: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


class TicketCreateRequest(BaseModel):
    subject: str
    description: str
    reporter: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "medium"


@app.post("/api/ticket")
async def create_ticket(req: TicketCreateRequest):
    """Create a manual IT support ticket (mock implementation)."""
    ticket_id = f"TKT_{uuid.uuid4().hex[:8].upper()}"
    # In a real system we'd persist this to a DB; here we return a mocked ticket
    ticket = {
        "ticket_id": ticket_id,
        "subject": req.subject,
        "description": req.description,
        "reporter": req.reporter,
        "priority": req.priority,
        "created_at": datetime.now().isoformat(),
        "status": "open"
    }
    logger.info(f"Created ticket: {ticket}")
    return ticket


@app.get("/api/ticket/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Lookup a ticket using the mocked ticket lookup tool."""
    result = lookup_ticket(ticket_id)
    return {"ticket_id": ticket_id, "result": result}

@app.get("/graph/mermaid")
async def get_mermaid_diagram():
    """Generate a Mermaid diagram for the workflow graph."""
    try:
        mermaid_syntax = support_graph.get_graph(xray=True).draw_mermaid()
        return {
            "mermaid_diagram": mermaid_syntax,
            "description": "Customer Support Agent Graph Flow"
        }
    except Exception as e:
        logger.error(f"Mermaid generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.get("/")
async def root():
    """Provide an overview of the API and its endpoints."""
    logger.debug("Entering root endpoint")
    response = {
        "message": "Customer Support Agent API",
        "endpoints": {
            "chat": "POST /api/chat",
            "feedback": "POST /api/feedback/{thread_id}",
            "status": "GET /api/status/{thread_id}",
            "conversation": "GET /api/conversation/{thread_id}"
        }
    }
    logger.info(f"Root endpoint response: {response}")
    logger.debug("Exiting root endpoint")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000, log_level="debug")