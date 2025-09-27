from langgraph.graph import StateGraph, END
import uuid
from typing_extensions import TypedDict
import random
from typing import Literal
from dotenv import load_dotenv
load_dotenv()


class MultiAgentState(TypedDict):
    task: str
    research_results: list
    analysis: str
    final_report: str
    active_agent: str
    conversation_history: list

def researcher_agent(state):
    # Simulate research
    research = f"Research findings for: {state['task']}"
    return {
        "research_results": state.get("research_results", []) + [research],
        "active_agent": "researcher",
        "conversation_history": state.get("conversation_history", []) + 
                               [f"Researcher: Completed research on {state['task']}"]
    }

def analyst_agent(state):
    # Analyze research results
    results = state.get("research_results", [])
    analysis = f"Analysis of {len(results)} research findings: Key insights discovered"
    return {
        "analysis": analysis,
        "active_agent": "analyst",
        "conversation_history": state.get("conversation_history", []) + 
                               [f"Analyst: Completed analysis"]
    }

def writer_agent(state):
    # Create final report
    report = f"Final Report: {state['task']}\n"
    report += f"Research: {len(state.get('research_results', []))} sources\n"
    report += f"Analysis: {state.get('analysis', '')}\n"
    
    return {
        "final_report": report,
        "active_agent": "writer",
        "conversation_history": state.get("conversation_history", []) + 
                               [f"Writer: Created final report"]
    }

def coordinator(state):
    # Decide next agent
    if not state.get("research_results"):
        return "researcher"
    elif not state.get("analysis"):
        return "analyst"
    elif not state.get("final_report"):
        return "writer"
    else:
        return "end"

# Multi-agent workflow
multi_agent_workflow = StateGraph(MultiAgentState)

multi_agent_workflow.add_node("researcher", researcher_agent)
multi_agent_workflow.add_node("analyst", analyst_agent)
multi_agent_workflow.add_node("writer", writer_agent)

# Coordinator decides routing
multi_agent_workflow.add_conditional_edges(
    "researcher",
    coordinator,
    {
        "analyst": "analyst",
        "end": END
    }
)

multi_agent_workflow.add_conditional_edges(
    "analyst",
    coordinator,
    {
        "writer": "writer",
        "end": END
    }
)

multi_agent_workflow.add_edge("writer", END)
multi_agent_workflow.set_entry_point("researcher")

multi_agent_app = multi_agent_workflow.compile()