from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from typing import Annotated, Sequence, TypedDict
from langchain.schema import BaseMessage, HumanMessage
import operator
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv
load_dotenv()

serper_api_wrapper = GoogleSerperAPIWrapper()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

# 1. Define the state for the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Create the different agents (nodes)
print("---"*10)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("No Gemini credentials found: set GOOGLE_API_KEY run gcloud auth application-default login")

# 2. Create the agent with a single line
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

def supervisor_node(state):
    # Logic to decide which agent to hand off to
    if "championships" in state["messages"][-1].content:
        return {"goto": "analytics_agent"}
    else:
        return {"goto": "research_agent"}

def research_agent(state):
    # Perform search using tools
    # Update state with findings
    return {"messages": ...}

def analytics_agent(state):
    # Perform a different search or calculation
    # Update state with findings
    return {"messages": ...}

# 3. Build the graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", research_agent)
graph.add_node("analytics_agent", analytics_agent)

# 4. Define the edges (transitions)
graph.add_edge("research_agent", "supervisor")
graph.add_edge("analytics_agent", END)

graph.add_conditional_edges(
    "supervisor",
    lambda state: state['goto'],
    {
        "analytics_agent": "analytics_agent", 
        "research_agent": "research_agent"
    }
)

graph.set_entry_point("supervisor")
runnable = graph.compile()


