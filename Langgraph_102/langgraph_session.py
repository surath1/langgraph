from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import Dict, List, Any, Literal
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
# ============== HELPDESK TOOLS ==============

@tool
def search_knowledge_base(query: str) -> str:
    """Search the helpdesk knowledge base for solutions to common problems."""
    # Mock knowledge base - in real app, this would query a database
    knowledge_base = {
        "password reset": "To reset your password: 1) Go to login page 2) Click 'Forgot Password' 3) Enter your email 4) Check email for reset link 5) Follow instructions in email",
        "email not working": "For email issues: 1) Check internet connection 2) Verify email settings 3) Clear email cache 4) Restart email client 5) Contact IT if problem persists",
        "software install": "To install software: 1) Check system requirements 2) Download from official source 3) Run as administrator 4) Follow installation wizard 5) Restart computer if required",
        "printer problem": "For printer issues: 1) Check power and connections 2) Verify paper and ink levels 3) Clear print queue 4) Update printer drivers 5) Restart printer and computer",
        "wifi connection": "WiFi troubleshooting: 1) Check if WiFi is enabled 2) Restart router and device 3) Forget and reconnect to network 4) Update network drivers 5) Contact network admin",
        "slow computer": "To fix slow computer: 1) Check available storage space 2) Close unnecessary programs 3) Run virus scan 4) Update system and drivers 5) Consider hardware upgrade"
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Knowledge Base Solution:\n{value}"
    
    return f"No specific solution found for '{query}'. Please provide more details or contact a technician for advanced support."

@tool
def create_ticket(title: str, description: str, priority: str = "Medium") -> str:
    """Create a support ticket for issues that need technician attention."""
    ticket_id = f"HELP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    # Mock ticket creation - in real app, this would save to database
    ticket_data = {
        "id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().isoformat()
    }
    
    return f"Support ticket created successfully!\nTicket ID: {ticket_id}\nTitle: {title}\nPriority: {priority}\nStatus: Open\n\nA technician will contact you within 24 hours for {priority.lower()} priority tickets."

@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of an existing support ticket."""
    # Mock ticket status - in real app, this would query database
    mock_tickets = {
        "HELP-20241201-ABC123": {"status": "In Progress", "assigned_to": "John Smith", "last_update": "2024-12-01 14:30"},
        "HELP-20241130-DEF456": {"status": "Resolved", "resolution": "Password reset completed", "resolved_at": "2024-11-30 16:45"}
    }
    
    if ticket_id in mock_tickets:
        ticket = mock_tickets[ticket_id]
        return f"Ticket {ticket_id}:\nStatus: {ticket['status']}\nAssigned to: {ticket.get('assigned_to', 'Unassigned')}\nLast Update: {ticket.get('last_update', 'N/A')}\nResolution: {ticket.get('resolution', 'N/A')}"
    
    return f"Ticket {ticket_id} not found. Please verify the ticket ID and try again."

@tool
def get_system_status() -> str:
    """Check current status of IT systems and services."""
    # Mock system status - in real app, this would check actual services
    systems = {
        "Email Server": "Online",
        "File Server": "Online", 
        "WiFi Network": "Online",
        "Printer Network": "Maintenance (Expected resolution: 3:00 PM)",
        "VPN Service": "Online",
        "Database": "Online"
    }
    
    status_report = "Current System Status:\n" + "="*25 + "\n"
    for system, status in systems.items():
        status_emoji = "🟢" if status == "Online" else "🟡" if "Maintenance" in status else "🔴"
        status_report += f"{status_emoji} {system}: {status}\n"
    
    return status_report

@tool
def escalate_to_human(reason: str, user_context: str = "") -> str:
    """Escalate the conversation to a human agent when the issue is complex or requires personal attention."""
    escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M')}"
    
    return f"Escalating to human agent...\nEscalation ID: {escalation_id}\nReason: {reason}\n\nA human agent will join this conversation within 5 minutes. Please stay connected and provide any additional details about your issue."

# ============== CHATBOT STATE & LOGIC ==============

class HelpdeskState(MessagesState):
    user_id: str = ""
    session_metadata: Dict[str, Any] = {}
    current_issue: str = ""
    escalated: bool = False

# Initialize components
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

# Create tools list
tools = [search_knowledge_base, create_ticket, check_ticket_status, get_system_status, escalate_to_human]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ============== GRAPH NODES ==============

def chatbot_node(state: HelpdeskState):
    """Main helpdesk assistant node"""
    messages = state["messages"]
    
    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content="""You are a professional IT Helpdesk Assistant. Your role is to:

1. **Help users solve technical problems** using the knowledge base and available tools
2. **Create support tickets** for complex issues that need technician attention
3. **Check system status** and ticket updates
4. **Escalate to human agents** when necessary

Guidelines:
- Be friendly, professional, and patient
- Ask clarifying questions to understand the issue
- Use tools proactively to find solutions
- Always try knowledge base search first for common problems
- Create tickets for hardware issues, complex software problems, or anything requiring hands-on support
- Escalate to humans for urgent issues, frustrated users, or when you can't help

Available tools:
- search_knowledge_base: Find solutions for common IT issues
- create_ticket: Create support tickets for technician attention
- check_ticket_status: Check existing ticket status
- get_system_status: Check if systems are operational
- escalate_to_human: Transfer to human agent

Start each conversation by greeting the user and asking how you can help with their IT issue today.""")
        messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: HelpdeskState) -> Literal["tools", "end"]:
    """Decide whether to use tools or end the conversation"""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"

# ============== BUILD GRAPH ==============

workflow = StateGraph(HelpdeskState)

# Add nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("chatbot")

# Add edges
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After using tools, go back to chatbot
workflow.add_edge("tools", "chatbot")

# Compile the graph
app = workflow.compile(checkpointer=memory)

# ============== SESSION MANAGER ==============

class HelpdeskSessionManager:
    def __init__(self, compiled_graph):
        self.graph = compiled_graph
        self.active_sessions = {}
    
    def create_session(self, user_id: str, user_name: str = None, department: str = None) -> str:
        """Create a new helpdesk session"""
        session_id = str(uuid.uuid4())
        
        session_metadata = {
            "user_id": user_id,
            "user_name": user_name or "User",
            "department": department or "Unknown",
            "created_at": datetime.now().isoformat(),
            "issue_count": 0
        }
        
        self.active_sessions[session_id] = session_metadata
        print(f"🆕 New helpdesk session created: {session_id}")
        print(f"👤 User: {user_name or user_id} ({department or 'Unknown dept.'})")
        
        return session_id
    
    def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send message to helpdesk assistant"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found. Please create a new session."}
        
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        try:
            human_message = HumanMessage(content=message)
            result = self.graph.invoke(
                {
                    "messages": [human_message],
                    "user_id": self.active_sessions[session_id]["user_id"],
                    "session_metadata": self.active_sessions[session_id]
                }, 
                config=config
            )
            
            # Update issue count
            self.active_sessions[session_id]["issue_count"] += 1
            
            return result
            
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id not in self.active_sessions:
            return []
        
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)
        
        if state.values and "messages" in state.values:
            return [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                }
                for msg in state.values["messages"]
                if not isinstance(msg, SystemMessage)  # Skip system messages in history
            ]
        return []
    
    def end_session(self, session_id: str) -> str:
        """End a helpdesk session"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            return f"Session ended for {session_info['user_name']}. Issues handled: {session_info['issue_count']}"
        return "Session not found."

# ============== USAGE EXAMPLE ==============

def main():
    # Initialize helpdesk
    helpdesk = HelpdeskSessionManager(app)
    
    # Create session for user
    session_id = helpdesk.create_session(
        user_id="emp001", 
        user_name="Alice Johnson", 
        department="Marketing"
    )
    
    print("\n" + "="*60)
    print("🎯 IT HELPDESK CHATBOT - LIVE DEMO")
    print("="*60)
    
    # Simulate conversation
    test_messages = [
        "Hi, I'm having trouble with my email. It's not receiving new messages.",
        "I checked my internet connection and it seems fine. What else can I try?",
        "The issue is still there after following those steps. Can you create a ticket?",
        "Also, can you check if the email server is having any issues?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n👤 User: {message}")
        print("🤖 Assistant: ", end="")
        
        response = helpdesk.send_message(session_id, message)
        
        if "error" in response:
            print(f"❌ {response['error']}")
        else:
            # Get the last AI message
            last_message = response["messages"][-1]
            print(last_message.content)
            
            # If there were tool calls, show what tools were used
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"\n🔧 Tools used: {[tool_call['name'] for tool_call in last_message.tool_calls]}")
        
        print("-" * 60)
    
    # Show session summary
    print(f"\n📊 Session Summary:")
    history = helpdesk.get_session_history(session_id)
    print(f"Total messages: {len(history)}")
    
    # End session
    print(f"\n🏁 {helpdesk.end_session(session_id)}")

if __name__ == "__main__":
    main()

# ============== ADVANCED USAGE INTERACTIVE ==============

# def interactive_demo():
#     """Run an interactive demo of the helpdesk chatbot"""
#     helpdesk = HelpdeskSessionManager(app)
    
#     print("🎯 IT HELPDESK CHATBOT - INTERACTIVE MODE")
#     print("Type 'quit' to exit, 'history' to see conversation history")
#     print("="*60)
    
#     # Get user info
#     user_name = input("Enter your name: ").strip() or "User"
#     department = input("Enter your department: ").strip() or "General"
    
#     session_id = helpdesk.create_session(
#         user_id=f"interactive_{uuid.uuid4().hex[:8]}", 
#         user_name=user_name, 
#         department=department
#     )
    
#     print(f"\n👋 Welcome {user_name}! How can I help you today?")
    
#     while True:
#         try:
#             user_input = input(f"\n{user_name}: ").strip()
            
#             if user_input.lower() == 'quit':
#                 print(helpdesk.end_session(session_id))
#                 break
#             elif user_input.lower() == 'history':
#                 history = helpdesk.get_session_history(session_id)
#                 print(f"\n📜 Conversation History ({len(history)} messages):")
#                 for msg in history[-10:]:  # Show last 10 messages
#                     print(f"  {msg['type']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
#                 continue
#             elif not user_input:
#                 continue
            
#             print("🤖 Assistant: ", end="", flush=True)
#             response = helpdesk.send_message(session_id, user_input)
            
#             if "error" in response:
#                 print(f"❌ {response['error']}")
#             else:
#                 print(response["messages"][-1].content)
                
#         except KeyboardInterrupt:
#             print(f"\n\n{helpdesk.end_session(session_id)}")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")

# # Uncomment to run interactive demo
# interactive_demo()