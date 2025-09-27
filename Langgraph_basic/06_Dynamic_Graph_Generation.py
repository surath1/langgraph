
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from IPython.display import Image, display
from dotenv import load_dotenv
load_dotenv()


def create_dynamic_workflow(steps: list):
    """Create a workflow dynamically based on input steps"""
    
    class DynamicState(TypedDict):
        current_step: int
        results: list
        step_data: dict
    
    def create_step_function(step_name, step_logic):
        def step_function(state):
            # Execute step logic
            result = step_logic(state.get("step_data", {}))
            return {
                "results": state.get("results", []) + [f"{step_name}: {result}"],
                "current_step": state.get("current_step", 0) + 1
            }
        return step_function
    
    workflow = StateGraph(DynamicState)
    
    # Add nodes dynamically
    for i, (step_name, step_logic) in enumerate(steps):
        node_name = f"step_{i}"
        workflow.add_node(node_name, create_step_function(step_name, step_logic))
        
        if i == 0:
            workflow.set_entry_point(node_name)
        else:
            prev_node = f"step_{i-1}"
            workflow.add_edge(prev_node, node_name)
    
    # Add final edge to END
    if steps:
        final_node = f"step_{len(steps)-1}"
        workflow.add_edge(final_node, END)
    
    return workflow.compile()

# Example usage
def process_data(data):
    return "processed"

def validate_data(data):
    return "validated"

def save_data(data):
    return "saved"

dynamic_steps = [
    ("Process", process_data),
    ("Validate", validate_data),
    ("Save", save_data)
]

dynamic_app = create_dynamic_workflow(dynamic_steps)