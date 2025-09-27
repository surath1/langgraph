from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict
import random
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

class HumanLoopState(TypedDict):
    user_input: str
    ai_response: str
    human_approval: str
    iteration_count: int

def ai_generate(state):
    # Simulate AI generating a response
    response = f"AI generated response to: {state['user_input']}"
    return {
        "ai_response": response,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def human_review(state):
    # This would normally pause for human input
    # For demo, we'll simulate different responses
    iteration = state.get("iteration_count", 0)
    
    if iteration == 1:
        return {"human_approval": "needs_revision"}
    else:
        return {"human_approval": "approved"}

def revise_response(state):
    revised = f"Revised: {state['ai_response']} (iteration {state['iteration_count']})"
    return {"ai_response": revised}

def should_revise(state):
    if state.get("human_approval") == "needs_revision":
        return "revise"
    else:
        return "end"

# Create workflow with checkpointing
memory = SqliteSaver.from_conn_string(":memory:")
human_loop_workflow = StateGraph(HumanLoopState)

human_loop_workflow.add_node("generate", ai_generate)
human_loop_workflow.add_node("human_review", human_review)
human_loop_workflow.add_node("revise", revise_response)

human_loop_workflow.add_edge("generate", "human_review")
human_loop_workflow.add_conditional_edges(
    "human_review",
    should_revise,
    {
        "revise": "revise",
        "end": END
    }
)
human_loop_workflow.add_edge("revise", "generate")

human_loop_workflow.set_entry_point("generate")
human_loop_app = human_loop_workflow.compile(checkpointer=memory)