import os
import operator
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict # Recommended for compatibility
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field

# -------------------- 1. Define tools --------------------
# Define the schema for your tool's input
class GoogleSearchInput(BaseModel):
    """Input for the Google Search tool."""
    query: str = Field(description="The search query to perform.")

# The rest of your code
@tool(args_schema=GoogleSearchInput)
def google_search(query: str) -> str:
    """Performs a Google search and returns the results."""
    print(f"\n--- Calling Google Search with: '{query}' ---\n")
    return "NA - total championships."

# -------------------- Define the graph state --------------------
class AgentState(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]

# -------------------- Configure the LLM and API key --------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("No Gemini credentials found: set GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

# -------------------- 2. Bind the tools to the LLM --------------------
llm_with_tools = llm.bind_tools([google_search])

# -------------------- 3. Create a ToolNode to execute tools ----------
tool_node = ToolNode([google_search])

# -------------------- 4. Define the agent nodes and functions ---------

def agent_node(state):
    """The node for the research and analytics agents."""
    messages = state["messages"]
    # The LLM with tools is called here
    return {"messages": [llm_with_tools.invoke(messages)]}

def supervisor_node(state):
    """
    Decides whether to route to the analytics_agent, research_agent, or tool_node.
    """
    last_message = state["messages"][-1].content
    if "championships" in last_message.lower():
        # A more complex heuristic to decide if a tool is needed, for example.
        return {"goto": "analytics_agent"}
    return {"goto": "research_agent"}

# You need a function to decide the flow after an agent's turn
def should_continue(state):
    """Conditional edge logic: decide if the agent should call a tool."""
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "continue" # Keep processing for tool calls
    else:
        # Check if the last message from the LLM contains a tool call
        if last_message.tool_calls:
            return "call_tool"
        return "end"

# The research and analytics agents are now just a single agent_node with routing
# The supervisor determines which "persona" is needed before the LLM is called.
# The tool call is handled automatically if the LLM suggests it.

# -------------------- 5. Build the graph --------------------
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("call_tool", tool_node)
graph.add_node("supervisor", supervisor_node) # Supervisor for initial routing logic

# Define the edges (transitions)
# Start the graph with the supervisor
graph.set_entry_point("supervisor")

# Route from the agent node to call the tool or end the conversation
graph.add_conditional_edges("agent_node", should_continue, {
    "call_tool": "call_tool",
    "end": END
})

# After the tool is called, the result is sent back to the agent for interpretation
graph.add_edge("call_tool", "agent_node")

# Route from the supervisor to the appropriate agent_node
graph.add_edge("supervisor", "agent_node")

# Compile the graph
runnable = graph.compile()

# Example invocation
result = runnable.invoke({"messages": [HumanMessage(content="Which team won the NBA finals and how many championships do they have?")]})
print(result["messages"][-1].content)
