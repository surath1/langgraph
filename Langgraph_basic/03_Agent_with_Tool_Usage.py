
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import tool
import os   
from typing_extensions import TypedDict
import random
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for '{query}': [Mock search results]"

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression)  # Note: Use safely in production
        return str(result)
    except:
        return "Error in calculation"

class AgentState(TypedDict):
    messages: list
    tool_calls: list
    tool_responses: list
    next_action: str

def agent_node(state):
    # Simulate LLM deciding on tool usage
    last_message = state["messages"][-1] if state["messages"] else ""
    
    if "search" in last_message.lower():
        tool_calls = [{"tool": "search_web", "args": {"query": "example query"}}]
        return {
            "tool_calls": tool_calls,
            "next_action": "use_tools"
        }
    elif any(char in last_message for char in "+-*/"):
        tool_calls = [{"tool": "calculate", "args": {"expression": last_message}}]
        return {
            "tool_calls": tool_calls,
            "next_action": "use_tools"
        }
    else:
        return {
            "messages": ["I understand your request"],
            "next_action": "end"
        }

def tool_node(state):
    tools = {"search_web": search_web, "calculate": calculate}
    responses = []
    
    for tool_call in state["tool_calls"]:
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]
        
        if tool_name in tools:
            result = tools[tool_name](**tool_args)
            responses.append(result)
    
    return {
        "tool_responses": responses,
        "next_action": "respond"
    }

def response_node(state):
    responses = state.get("tool_responses", [])
    response = f"Based on the tool results: {', '.join(responses)}"
    
    return {
        "messages": state["messages"] + [response],
        "next_action": "end"
    }

def should_use_tools(state):
    return state.get("next_action", "") == "use_tools"

def should_respond(state):
    return state.get("next_action", "") == "respond"

# Build agent graph
agent_workflow = StateGraph(AgentState)

agent_workflow.add_node("agent", agent_node)
agent_workflow.add_node("tools", tool_node)
agent_workflow.add_node("respond", response_node)

agent_workflow.add_conditional_edges(
    "agent",
    lambda state: state.get("next_action", "end"),
    {
        "use_tools": "tools",
        "end": END
    }
)

agent_workflow.add_conditional_edges(
    "tools",
    lambda state: state.get("next_action", "end"),
    {
        "respond": "respond",
        "end": END
    }
)

agent_workflow.add_edge("respond", END)
agent_workflow.set_entry_point("agent")

agent_app = agent_workflow.compile()