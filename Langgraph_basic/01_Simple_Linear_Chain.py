from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from IPython.display import Image, display

# Define the state structure
class GraphState(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str

def step_1(state):
    return {
        "messages": ["Step 1 completed"],
        "current_step": "step_1"
    }

def step_2(state):
    return {
        "messages": ["Step 2 completed"],
        "current_step": "step_2"
    }

# Create the graph
workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("step1", step_1)
workflow.add_node("step2", step_2)

# Add edges
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

# Set entry point
workflow.set_entry_point("step1")

# Compile the graph
app = workflow.compile()

# Execute
result = app.invoke({"messages": [
    {
        "messages": ["Step 0 started"],
        "current_step": "step_0"
    }
], "current_step": ""})
#print(result["messages"])
for i in result["messages"]:
    print(i)

print("--------------------------------")

display(Image(app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))