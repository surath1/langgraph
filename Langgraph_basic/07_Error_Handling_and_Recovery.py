from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from IPython.display import Image, display
from dotenv import load_dotenv
load_dotenv()


class RobustState(TypedDict):
    task: str
    attempts: int
    errors: list
    success: bool
    result: str

def risky_operation(state):
    attempts = state.get("attempts", 0) + 1
    
    # Simulate failure on first two attempts
    if attempts < 3:
        error = f"Attempt {attempts} failed"
        return {
            "attempts": attempts,
            "errors": state.get("errors", []) + [error],
            "success": False
        }
    else:
        return {
            "attempts": attempts,
            "success": True,
            "result": f"Success after {attempts} attempts"
        }

def error_handler(state):
    errors = state.get("errors", [])
    return {
        "result": f"Failed after {len(errors)} errors: {'; '.join(errors[-3:])}"
    }

def should_retry(state):
    if state.get("success"):
        return "end"
    elif state.get("attempts", 0) < 3:
        return "retry"
    else:
        return "handle_error"

# Robust workflow with error handling
robust_workflow = StateGraph(RobustState)

robust_workflow.add_node("operation", risky_operation)
robust_workflow.add_node("error_handler", error_handler)

robust_workflow.add_conditional_edges(
    "operation",
    should_retry,
    {
        "retry": "operation",
        "handle_error": "error_handler",
        "end": END
    }
)

robust_workflow.add_edge("error_handler", END)
robust_workflow.set_entry_point("operation")

robust_app = robust_workflow.compile()