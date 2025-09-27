from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from IPython.display import Image, display
from dotenv import load_dotenv
load_dotenv()

class CustomerServiceState(TypedDict):
    customer_query: str
    intent: str
    customer_data: dict
    resolution_steps: list
    satisfaction_score: int
    escalate: bool

def intent_classification(state):
    query = state["customer_query"].lower()
    
    if "refund" in query or "return" in query:
        intent = "refund_request"
    elif "track" in query or "order" in query:
        intent = "order_tracking"
    elif "cancel" in query:
        intent = "cancellation"
    else:
        intent = "general_inquiry"
    
    return {"intent": intent}

def fetch_customer_data(state):
    # Simulate database lookup
    return {"customer_data": {"tier": "premium", "orders": 5}}

def handle_refund(state):
    steps = ["Verify purchase", "Check refund policy", "Process refund"]
    return {"resolution_steps": steps}

def handle_tracking(state):
    steps = ["Look up order", "Provide tracking info"]
    return {"resolution_steps": steps}

def handle_cancellation(state):
    steps = ["Verify order", "Cancel if possible", "Confirm cancellation"]
    return {"resolution_steps": steps}

def general_support(state):
    steps = ["Understand issue", "Provide solution"]
    return {"resolution_steps": steps}

def quality_check(state):
    # Simulate quality assessment
    return {"satisfaction_score": 8}

def should_escalate(state):
    return state.get("satisfaction_score", 0) < 6

def route_by_intent(state):
    intent_map = {
        "refund_request": "handle_refund",
        "order_tracking": "handle_tracking", 
        "cancellation": "handle_cancellation",
        "general_inquiry": "general_support"
    }
    return intent_map.get(state.get("intent"), "general_support")

# Customer service workflow
cs_workflow = StateGraph(CustomerServiceState)

cs_workflow.add_node("classify_intent", intent_classification)
cs_workflow.add_node("fetch_data", fetch_customer_data)
cs_workflow.add_node("handle_refund", handle_refund)
cs_workflow.add_node("handle_tracking", handle_tracking)
cs_workflow.add_node("handle_cancellation", handle_cancellation)
cs_workflow.add_node("general_support", general_support)
cs_workflow.add_node("quality_check", quality_check)

cs_workflow.add_edge("classify_intent", "fetch_data")
cs_workflow.add_conditional_edges(
    "fetch_data",
    route_by_intent,
    {
        "handle_refund": "handle_refund",
        "handle_tracking": "handle_tracking",
        "handle_cancellation": "handle_cancellation",
        "general_support": "general_support"
    }
)

for handler in ["handle_refund", "handle_tracking", "handle_cancellation", "general_support"]:
    cs_workflow.add_edge(handler, "quality_check")

cs_workflow.add_conditional_edges(
    "quality_check",
    lambda state: "escalate" if should_escalate(state) else "end",
    {
        "escalate": END,  # Would route to human agent
        "end": END
    }
)

cs_workflow.set_entry_point("classify_intent")
cs_app = cs_workflow.compile()