from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import os   
from typing_extensions import TypedDict
import random
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

#define States
class NodeState(TypedDict):
    graph_state: str


#decide_which_node
def decide_which_node(state) -> Literal["node2", "node3"]:
    user_input = state['graph_state']
    if random.random() < 0.5:
        return "node2"
    return "node3"


def node_1(state):
    print("---Node 1---")
    tmp_state = state['graph_state'] +" Hello Surath"
    return {"graph_state": tmp_state}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" not happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" supper happy"}


# Build graph
builder = StateGraph(NodeState)
builder.add_node("node1", node_1)
builder.add_node("node2", node_2)
builder.add_node("node3", node_3)

# Logic
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_which_node)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

