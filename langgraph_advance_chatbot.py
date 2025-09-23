from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import Dict, List, Any, Literal, Optional
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import smtplib
from email.mime.text import MIMEText
import requests
import logging
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
load_dotenv()
# ============== ADVANCED DATA MODELS ==============

class Priority(Enum):
    CRITICAL = "Critical"
    HIGH = "High" 
    MEDIUM = "Medium"
    LOW = "Low"

class TicketStatus(Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    PENDING = "Pending User Response"
    RESOLVED = "Resolved"
    CLOSED = "Closed"

class SystemStatus(Enum):
    OPERATIONAL = "Operational"
    DEGRADED = "Degraded Performance"
    MAINTENANCE = "Scheduled Maintenance"
    OUTAGE = "Service Outage"

@dataclass
class User:
    user_id: str
    name: str
    email: str
    department: str
    role: str
    location: str
    manager: str
    phone: Optional[str] = None
    vip: bool = False

@dataclass
class Ticket:
    ticket_id: str
    user_id: str
    title: str
    description: str
    priority: Priority
    status: TicketStatus
    category: str
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolution: Optional[str] = None
    resolution_time: Optional[int] = None  # minutes
    escalation_level: int = 0
    attachments: List[str] = field(default_factory=list)

# ============== ADVANCED TOOLS ==============

@tool
def search_advanced_knowledge_base(query: str, category: str = "general") -> str:
    """Search advanced knowledge base with categorized solutions and step-by-step guides."""
    
    knowledge_base = {
        "hardware": {
            "computer slow": {
                "solution": "Advanced Computer Performance Optimization",
                "steps": [
                    "1. Check CPU usage in Task Manager (Ctrl+Shift+Esc)",
                    "2. Identify resource-heavy processes and end unnecessary ones",
                    "3. Check available disk space (minimum 15% free required)",
                    "4. Run Disk Cleanup (cleanmgr) and Disk Defragmenter",
                    "5. Check for malware using Windows Defender full scan",
                    "6. Update device drivers through Device Manager",
                    "7. Disable startup programs via Task Manager > Startup tab",
                    "8. Check RAM usage - upgrade if consistently >80%",
                    "9. Monitor temperatures - clean fans if overheating"
                ],
                "estimated_time": "30-45 minutes",
                "difficulty": "Intermediate"
            },
            "printer issues": {
                "solution": "Comprehensive Printer Troubleshooting Protocol",
                "steps": [
                    "1. Verify printer power and connection status",
                    "2. Check paper tray and ink/toner levels",
                    "3. Clear print queue: Settings > Devices > Printers",
                    "4. Run Windows printer troubleshooter",
                    "5. Update printer driver from manufacturer website",
                    "6. Reset printer spooler service (net stop spooler, net start spooler)",
                    "7. Check network connectivity for network printers",
                    "8. Print test page to isolate hardware issues",
                    "9. Check for firmware updates"
                ],
                "estimated_time": "20-30 minutes",
                "difficulty": "Beginner"
            }
        },
        "software": {
            "email problems": {
                "solution": "Enterprise Email Troubleshooting Guide",
                "steps": [
                    "1. Verify internet connectivity and DNS resolution",
                    "2. Check email server settings (IMAP/POP3/Exchange)",
                    "3. Clear Outlook cache and rebuild search index",
                    "4. Create new Outlook profile if corruption suspected",
                    "5. Check mailbox storage quota and archive old emails",
                    "6. Verify firewall/antivirus email scanning settings",
                    "7. Test with Outlook Safe Mode (outlook.exe /safe)",
                    "8. Check for Exchange server connectivity issues",
                    "9. Rebuild OST file if synchronization errors occur"
                ],
                "estimated_time": "25-40 minutes",
                "difficulty": "Advanced"
            },
            "software installation": {
                "solution": "Enterprise Software Deployment Protocol",
                "steps": [
                    "1. Verify software compatibility with current OS version",
                    "2. Check system requirements (RAM, disk space, dependencies)",
                    "3. Download software from authorized/approved sources only",
                    "4. Run installation as administrator",
                    "5. Temporarily disable antivirus during installation",
                    "6. Follow corporate software installation policies",
                    "7. Register software with IT asset management system",
                    "8. Configure software according to company standards",
                    "9. Test functionality and create user documentation"
                ],
                "estimated_time": "15-60 minutes",
                "difficulty": "Intermediate"
            }
        },
        "security": {
            "password issues": {
                "solution": "Enterprise Password Management Protocol",
                "steps": [
                    "1. Attempt account unlock via self-service portal",
                    "2. Verify identity using secondary authentication",
                    "3. Check password policy requirements",
                    "4. Reset password using approved secure channel",
                    "5. Update password in all connected applications",
                    "6. Clear cached credentials (Windows Credential Manager)",
                    "7. Test access to all required systems",
                    "8. Enable MFA if not already configured",
                    "9. Document password change in compliance log"
                ],
                "estimated_time": "10-20 minutes",
                "difficulty": "Beginner"
            }
        },
        "network": {
            "connectivity issues": {
                "solution": "Advanced Network Connectivity Diagnostics",
                "steps": [
                    "1. Check physical network connections and indicators",
                    "2. Run ipconfig /release && ipconfig /renew",
                    "3. Flush DNS cache: ipconfig /flushdns",
                    "4. Test connectivity: ping 8.8.8.8 and ping domain.com",
                    "5. Check network adapter settings and drivers",
                    "6. Reset TCP/IP stack: netsh int ip reset",
                    "7. Verify firewall and VPN settings",
                    "8. Test with different DNS servers (1.1.1.1, 8.8.8.8)",
                    "9. Contact network team for switch/router issues"
                ],
                "estimated_time": "20-35 minutes",
                "difficulty": "Advanced"
            }
        }
    }
    
    # Search logic
    query_lower = query.lower()
    category_lower = category.lower()
    
    # Look in specific category first
    if category_lower in knowledge_base:
        for key, solution in knowledge_base[category_lower].items():
            if any(word in query_lower for word in key.split()):
                steps_text = "\n".join(solution["steps"])
                return f"""📋 **{solution['solution']}**
                
**Category**: {category.title()}
**Estimated Time**: {solution['estimated_time']}
**Difficulty**: {solution['difficulty']}

**Step-by-Step Instructions**:
{steps_text}

**Need Help?** If these steps don't resolve the issue, I can create a priority ticket for technical support."""
    
    # Search all categories
    for cat, items in knowledge_base.items():
        for key, solution in items.items():
            if any(word in query_lower for word in key.split()):
                steps_text = "\n".join(solution["steps"])
                return f"""📋 **{solution['solution']}**
                
**Category**: {cat.title()}
**Estimated Time**: {solution['estimated_time']}
**Difficulty**: {solution['difficulty']}

**Step-by-Step Instructions**:
{steps_text}

**Need Help?** If these steps don't resolve the issue, I can create a priority ticket for technical support."""
    
    return f"❓ No specific solution found for '{query}' in category '{category}'. I can help you create a detailed support ticket or escalate to our technical experts."

@tool
def create_advanced_ticket(title: str, description: str, priority: str = "Medium", 
                          category: str = "General", user_impact: str = "", 
                          business_impact: str = "", steps_attempted: str = "") -> str:
    """Create detailed support ticket with advanced categorization and SLA tracking."""
    
    ticket_id = f"HELP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    # Determine SLA based on priority
    sla_hours = {"Critical": 1, "High": 4, "Medium": 24, "Low": 72}
    response_time = sla_hours.get(priority, 24)
    
    # Calculate priority score based on impact
    priority_score = calculate_priority_score(priority, user_impact, business_impact)
    
    ticket_data = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "category": category,
        "user_impact": user_impact,
        "business_impact": business_impact,
        "steps_attempted": steps_attempted,
        "priority_score": priority_score,
        "status": "Open",
        "sla_response_hours": response_time,
        "created_at": datetime.now().isoformat(),
        "escalation_level": 0
    }
    
    # Auto-assign based on category and priority
    assigned_to = auto_assign_ticket(category, priority)
    ticket_data["assigned_to"] = assigned_to
    
    # Send notifications
    send_ticket_notifications(ticket_data)
    
    return f"""🎫 **Support Ticket Created Successfully**

**Ticket ID**: {ticket_id}
**Priority**: {priority} (Score: {priority_score})
**Category**: {category}
**Assigned To**: {assigned_to}
**SLA Response Time**: {response_time} hours

**Details**:
- Title: {title}
- User Impact: {user_impact or 'Not specified'}
- Business Impact: {business_impact or 'Not specified'}
- Steps Already Attempted: {steps_attempted or 'None specified'}

**Next Steps**:
✅ Ticket logged in IT Service Management system
✅ Assigned technician notified
✅ You'll receive email confirmation shortly
✅ Updates will be sent as work progresses

**Tracking**: You can check status anytime using ticket ID {ticket_id}"""

def calculate_priority_score(priority: str, user_impact: str, business_impact: str) -> int:
    """Calculate numerical priority score for ticket routing"""
    base_scores = {"Critical": 100, "High": 75, "Medium": 50, "Low": 25}
    score = base_scores.get(priority, 50)
    
    # Adjust based on impact descriptions
    if "multiple users" in user_impact.lower():
        score += 20
    if "business critical" in business_impact.lower():
        score += 25
    if "revenue impact" in business_impact.lower():
        score += 30
    
    return min(score, 100)

def auto_assign_ticket(category: str, priority: str) -> str:
    """Auto-assign tickets based on category and current workload"""
    # Simplified assignment logic - in real system would check actual workloads
    assignments = {
        "hardware": {"Critical": "Senior Tech - Mike Chen", "High": "Tech Lead - Sarah Wilson", 
                    "Medium": "Technician - Alex Rodriguez", "Low": "Junior Tech - Emma Davis"},
        "software": {"Critical": "Software Specialist - David Park", "High": "Senior Tech - Lisa Zhang",
                    "Medium": "Software Tech - Jordan Smith", "Low": "Junior Tech - Emma Davis"},
        "network": {"Critical": "Network Engineer - Tom Johnson", "High": "Network Specialist - Amy Chen",
                   "Medium": "Network Tech - Chris Anderson", "Low": "Junior Tech - Sam Wilson"},
        "security": {"Critical": "Security Lead - Jennifer Kim", "High": "Security Analyst - Mark Brown",
                    "Medium": "Security Tech - Rachel Green", "Low": "Security Intern - Kevin Lee"}
    }
    
    return assignments.get(category, {}).get(priority, "IT Support Team")

def send_ticket_notifications(ticket_data: dict):
    """Send notifications for new tickets (mock implementation)"""
    # In real implementation, would send actual emails/Slack/Teams notifications
    print(f"📧 Notification sent to {ticket_data['assigned_to']} for ticket {ticket_data['ticket_id']}")
    return True

@tool
def check_advanced_ticket_status(ticket_id: str) -> str:
    """Check detailed ticket status with timeline and next actions."""
    
    # Mock advanced ticket data - in real app would query database
    mock_tickets = {
        "HELP-20241201-ABC12345": {
            "status": "In Progress",
            "priority": "High",
            "category": "Hardware",
            "assigned_to": "Senior Tech - Mike Chen",
            "created_at": "2024-12-01 09:15:00",
            "updated_at": "2024-12-01 14:30:00",
            "progress": 60,
            "sla_deadline": "2024-12-01 17:15:00",
            "timeline": [
                {"time": "2024-12-01 09:15", "action": "Ticket created", "by": "System"},
                {"time": "2024-12-01 09:20", "action": "Assigned to Mike Chen", "by": "Auto-Assignment"},
                {"time": "2024-12-01 10:45", "action": "Initial diagnosis completed", "by": "Mike Chen"},
                {"time": "2024-12-01 14:30", "action": "Ordered replacement parts", "by": "Mike Chen"}
            ],
            "next_action": "Waiting for parts delivery (ETA: 2 hours)",
            "resolution_estimate": "2024-12-01 18:00:00"
        }
    }
    
    if ticket_id in mock_tickets:
        ticket = mock_tickets[ticket_id]
        
        # Calculate time metrics
        created = datetime.fromisoformat(ticket["created_at"])
        updated = datetime.fromisoformat(ticket["updated_at"])
        sla_deadline = datetime.fromisoformat(ticket["sla_deadline"])
        
        age_hours = (datetime.now() - created).total_seconds() / 3600
        time_to_sla = (sla_deadline - datetime.now()).total_seconds() / 3600
        
        timeline_str = "\n".join([
            f"  • {entry['time']} - {entry['action']} (by {entry['by']})"
            for entry in ticket["timeline"]
        ])
        
        status_emoji = {"Open": "🟡", "In Progress": "🔵", "Resolved": "🟢", "Closed": "⚫"}
        
        return f"""{status_emoji.get(ticket['status'], '🔵')} **Ticket Status: {ticket['status']}**

**Ticket ID**: {ticket_id}
**Priority**: {ticket['priority']} | **Category**: {ticket['category']}
**Assigned To**: {ticket['assigned_to']}
**Progress**: {ticket['progress']}% complete

**Timing**:
- Created: {age_hours:.1f} hours ago
- Last Updated: {(datetime.now() - updated).total_seconds()/3600:.1f} hours ago
- SLA Deadline: {time_to_sla:.1f} hours remaining
- Est. Resolution: {ticket.get('resolution_estimate', 'TBD')}

**Timeline**:
{timeline_str}

**Current Status**: {ticket['next_action']}

**Need Updates?** Your assigned technician will contact you with progress updates."""
    
    return f"❓ Ticket {ticket_id} not found. Please verify the ticket ID or contact support if you believe this is an error."

@tool
def get_comprehensive_system_status() -> str:
    """Get detailed system status with performance metrics and maintenance schedules."""
    
    systems = {
        "Email Services (Exchange)": {
            "status": "Operational",
            "uptime": "99.9%",
            "response_time": "125ms",
            "last_incident": "None",
            "maintenance_window": "Sunday 2:00-4:00 AM"
        },
        "File Server (SharePoint)": {
            "status": "Operational", 
            "uptime": "99.8%",
            "response_time": "200ms",
            "storage_used": "78%",
            "maintenance_window": "Saturday 11:00 PM - 1:00 AM"
        },
        "WiFi Network": {
            "status": "Operational",
            "uptime": "99.5%",
            "coverage": "98%",
            "peak_usage": "85% capacity",
            "maintenance_window": "As needed"
        },
        "Print Services": {
            "status": "Degraded Performance",
            "uptime": "94.2%",
            "issue": "3 printers offline for maintenance",
            "affected_locations": ["Floor 3 East", "Floor 5 West"],
            "eta_resolution": "December 1, 3:00 PM"
        },
        "VPN Services": {
            "status": "Operational",
            "uptime": "99.9%",
            "active_connections": "234/500",
            "response_time": "180ms",
            "maintenance_window": "Monthly, 1st Sunday 1:00-3:00 AM"
        },
        "Database Systems": {
            "status": "Operational",
            "uptime": "99.99%",
            "response_time": "45ms",
            "backup_status": "All backups current",
            "maintenance_window": "Sunday 12:00-2:00 AM"
        }
    }
    
    status_report = "🖥️ **COMPREHENSIVE SYSTEM STATUS DASHBOARD**\n"
    status_report += "=" * 55 + "\n\n"
    
    operational_count = 0
    total_systems = len(systems)
    
    for system, details in systems.items():
        status = details["status"]
        
        if status == "Operational":
            emoji = "🟢"
            operational_count += 1
        elif status == "Degraded Performance":
            emoji = "🟡"
        elif "Maintenance" in status:
            emoji = "🔵"
        else:
            emoji = "🔴"
        
        status_report += f"{emoji} **{system}**\n"
        status_report += f"   Status: {status}\n"
        
        if "uptime" in details:
            status_report += f"   Uptime: {details['uptime']}\n"
        if "response_time" in details:
            status_report += f"   Response Time: {details['response_time']}\n"
        if "issue" in details:
            status_report += f"   ⚠️ Issue: {details['issue']}\n"
        if "eta_resolution" in details:
            status_report += f"   ETA: {details['eta_resolution']}\n"
        if "maintenance_window" in details:
            status_report += f"   Maintenance: {details['maintenance_window']}\n"
        
        status_report += "\n"
    
    # Overall health summary
    health_percentage = (operational_count / total_systems) * 100
    status_report += f"📊 **OVERALL SYSTEM HEALTH: {health_percentage:.1f}%**\n"
    status_report += f"✅ {operational_count}/{total_systems} systems fully operational\n\n"
    
    # Upcoming maintenance
    status_report += "🔧 **UPCOMING MAINTENANCE**:\n"
    status_report += "• Email Services: This Sunday 2:00-4:00 AM\n"
    status_report += "• Database Systems: This Sunday 12:00-2:00 AM\n"
    status_report += "• File Server: This Saturday 11:00 PM - 1:00 AM\n"
    
    return status_report

@tool
def escalate_to_human_agent(reason: str, urgency: str = "Normal", user_context: str = "", 
                           issue_complexity: str = "", previous_attempts: str = "") -> str:
    """Advanced escalation to human agents with detailed context and routing."""
    
    escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Determine escalation path based on reason and urgency
    escalation_routes = {
        "technical_complexity": "Senior Technical Specialist",
        "security_incident": "Security Operations Center",
        "vip_user": "VIP Support Team",
        "business_critical": "Incident Response Team",
        "customer_satisfaction": "Customer Success Manager",
        "policy_exception": "IT Management Team"
    }
    
    route_to = escalation_routes.get(reason.lower().replace(" ", "_"), "General Support Supervisor")
    
    # Estimate response time based on urgency
    response_times = {
        "Critical": "5 minutes",
        "High": "15 minutes", 
        "Normal": "30 minutes",
        "Low": "2 hours"
    }
    
    eta = response_times.get(urgency, "30 minutes")
    
    escalation_data = {
        "escalation_id": escalation_id,
        "reason": reason,
        "urgency": urgency,
        "route_to": route_to,
        "user_context": user_context,
        "issue_complexity": issue_complexity,
        "previous_attempts": previous_attempts,
        "created_at": datetime.now().isoformat()
    }
    
    # Send escalation notifications (mock)
    send_escalation_notifications(escalation_data)
    
    return f"""🚀 **ESCALATION INITIATED**

**Escalation ID**: {escalation_id}
**Urgency Level**: {urgency}
**Reason**: {reason}
**Routed To**: {route_to}
**Expected Response**: Within {eta}

**Context Provided**:
- Issue Complexity: {issue_complexity or 'Standard'}
- Previous Attempts: {previous_attempts or 'Initial contact'}
- User Context: {user_context or 'Standard user'}

**What Happens Next**:
✅ {route_to} has been notified
✅ Your case is now priority queued
✅ You'll receive immediate confirmation email
✅ A specialist will contact you within {eta}
✅ This conversation will remain accessible to the agent

**Emergency Contact**: If this is a critical business outage, call our emergency line: (555) HELP-NOW

Please stay available for the next {eta} for immediate assistance."""

def send_escalation_notifications(escalation_data: dict):
    """Send escalation notifications to appropriate teams"""
    print(f"🚨 ESCALATION ALERT sent to {escalation_data['route_to']} - ID: {escalation_data['escalation_id']}")
    return True

@tool
def analyze_user_sentiment_and_satisfaction(conversation_history: str, current_message: str = "") -> str:
    """Analyze user sentiment and satisfaction levels to proactively improve service."""
    
    # Simple sentiment analysis (in real app would use NLP models)
    negative_indicators = ['frustrated', 'angry', 'terrible', 'horrible', 'useless', 'waste', 'stupid']
    positive_indicators = ['great', 'excellent', 'perfect', 'amazing', 'helpful', 'thanks', 'solved']
    urgency_indicators = ['urgent', 'immediately', 'asap', 'critical', 'emergency', 'down', 'broken']
    
    text_to_analyze = (conversation_history + " " + current_message).lower()
    
    negative_score = sum(1 for word in negative_indicators if word in text_to_analyze)
    positive_score = sum(1 for word in positive_indicators if word in text_to_analyze)
    urgency_score = sum(1 for word in urgency_indicators if word in text_to_analyze)
    
    if negative_score > positive_score and negative_score >= 2:
        sentiment = "Negative - User appears frustrated"
        recommendation = "Consider escalation or additional attention"
    elif positive_score > negative_score:
        sentiment = "Positive - User seems satisfied"
        recommendation = "Continue current approach"
    else:
        sentiment = "Neutral - Standard interaction"
        recommendation = "Monitor for changes"
    
    urgency_level = "High" if urgency_score >= 2 else "Medium" if urgency_score >= 1 else "Normal"
    
    return f"""📊 **USER SENTIMENT ANALYSIS**

**Current Sentiment**: {sentiment}
**Urgency Level**: {urgency_level}
**Recommendation**: {recommendation}

**Metrics**:
- Negative Indicators: {negative_score}
- Positive Indicators: {positive_score}  
- Urgency Indicators: {urgency_score}

**Suggested Actions**:
{get_sentiment_based_actions(sentiment, urgency_level)}"""

def get_sentiment_based_actions(sentiment: str, urgency: str) -> str:
    """Get recommended actions based on sentiment analysis"""
    if "Negative" in sentiment:
        if urgency == "High":
            return "• Immediate escalation recommended\n• Offer direct phone support\n• Consider supervisor involvement"
        else:
            return "• Acknowledge user's frustration\n• Provide extra detailed explanations\n• Follow up proactively"
    elif "Positive" in sentiment:
        return "• Continue excellent service\n• Consider asking for feedback\n• Document successful resolution approach"
    else:
        return "• Maintain professional service level\n• Monitor for sentiment changes\n• Standard resolution process"

@tool
def create_knowledge_article(title: str, problem_description: str, solution_steps: str, 
                           category: str = "General", difficulty: str = "Medium") -> str:
    """Create new knowledge base article from resolved issues for future reference."""
    
    article_id = f"KB-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    article_data = {
        "article_id": article_id,
        "title": title,
        "problem_description": problem_description,
        "solution_steps": solution_steps,
        "category": category,
        "difficulty": difficulty,
        "created_at": datetime.now().isoformat(),
        "created_by": "AI Assistant",
        "review_status": "Pending Review",
        "usage_count": 0
    }
    
    return f"""📝 **KNOWLEDGE ARTICLE CREATED**

**Article ID**: {article_id}
**Title**: {title}
**Category**: {category}
**Difficulty**: {difficulty}

**Content Created**:
✅ Problem description documented
✅ Solution steps formatted
✅ Categorized for easy search
✅ Queued for expert review

**Next Steps**:
- Article will be reviewed by subject matter expert
- Once approved, it will be available in knowledge base
- Future users with similar issues will benefit from this solution

This helps us continuously improve our support quality!"""

# ============== ADVANCED STATE MANAGEMENT ==============

class AdvancedHelpdeskState(MessagesState):
    user_id: str = ""
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    current_issue: str = ""
    issue_category: str = ""
    priority_level: str = "Medium"
    escalated: bool = False
    satisfaction_score: int = 0
    resolution_attempted: bool = False
    tickets_created: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    conversation_stage: str = "initial"  # initial, troubleshooting, resolved, escalated
    user_sentiment: str = "neutral"
    vip_user: bool = False

# Initialize advanced components
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Create comprehensive tools list
tools = [
    search_advanced_knowledge_base,
    create_advanced_ticket, 
    check_advanced_ticket_status,
    get_comprehensive_system_status,
    escalate_to_human_agent,
    analyze_user_sentiment_and_satisfaction,
    create_knowledge_article
]

tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# ============== ADVANCED GRAPH NODES ==============

def advanced_chatbot_node(state: AdvancedHelpdeskState):
    """Advanced helpdesk assistant with context awareness and intelligent routing"""
    messages = state["messages"]
    user_sentiment = state.get("user_sentiment", "neutral")
    conversation_stage = state.get("conversation_stage", "initial")
    vip_user = state.get("vip_user", False)
    
    # Enhanced system message based on context
    if not messages or not isinstance(messages[0], SystemMessage):
        vip_note = " This is a VIP user - provide premium white-glove service." if vip_user else ""
        sentiment_note = f" Current user sentiment: {user_sentiment} - adjust approach accordingly." if user_sentiment != "neutral" else ""
        
        system_msg = SystemMessage(content=f"""You are an Enterprise IT Helpdesk Assistant with advanced capabilities. 

**Your Role & Capabilities**:
• Provide expert-level technical support with step-by-step guidance
• Use advanced tools for comprehensive problem diagnosis and resolution
• Create detailed tickets with proper categorization and priority scoring
• Analyze user sentiment and escalate proactively when needed
• Build knowledge articles from successful resolutions
• Handle complex enterprise IT scenarios across all technology domains

**Current Context**:
• Conversation Stage: {conversation_stage}
• User Sentiment: {user_sentiment}{vip_note}{sentiment_note}

**Service Excellence Standards**:
• Always search knowledge base first for comprehensive solutions
• Gather complete problem context before creating tickets
• Escalate proactively for frustrated users or complex issues
• Follow up on ticket status and provide detailed updates
• Create knowledge articles for novel solutions
• Maintain professional, empathetic communication

**Available Advanced Tools**:
• search_advanced_knowledge_base: Comprehensive solutions with step-by-step guides
• create_advanced_ticket: Detailed tickets with SLA tracking and auto-assignment
• check_advanced_ticket_status: Full timeline and progress tracking
• get_comprehensive_system_status: Detailed infrastructure health monitoring
• escalate_to_human_agent: Smart routing to appropriate specialists
• analyze_user_sentiment_and_satisfaction: Proactive service quality monitoring
• create_knowledge_article: Document solutions for future use

Begin each new conversation by warmly greeting the user and asking for details about their technical issue.""")
        
        messages = [system_msg] + messages
    
    # Analyze sentiment in current context
    conversation_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
    
    response = llm_with_tools.invoke(messages)
    
    # Update conversation stage based on response
    new_stage = determine_conversation_stage(response, state)
    
    return {
        "messages": [response],
        "conversation_stage": new_stage,
        "tools_used": state.get("tools_used", [])
    }

def determine_conversation_stage(response, state) -> str:
    """Determine current conversation stage for context tracking"""
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [call['name'] for call in response.tool_calls]
        
        if 'escalate_to_human_agent' in tool_names:
            return "escalated"
        elif 'create_advanced_ticket' in tool_names:
            return "ticket_created"
        elif 'search_advanced_knowledge_base' in tool_names:
            return "troubleshooting"
        elif 'create_knowledge_article' in tool_names:
            return "resolved"
    
    return state.get("conversation_stage", "initial")

def sentiment_analysis_node(state: AdvancedHelpdeskState):
    """Dedicated node for sentiment analysis and proactive escalation"""
    messages = state["messages"]
    
    # Get recent conversation for analysis
    recent_messages = messages[-5:] if len(messages) >= 5 else messages
    conversation_text = " ".join([
        msg.content for msg in recent_messages 
        if hasattr(msg, 'content') and isinstance(msg, (HumanMessage, AIMessage))
    ])
    
    # Simple sentiment scoring
    negative_words = ['frustrated', 'angry', 'terrible', 'horrible', 'useless', 'broken', 'failed', 'worst']
    positive_words = ['great', 'excellent', 'perfect', 'amazing', 'helpful', 'solved', 'working', 'thanks']
    
    text_lower = conversation_text.lower()
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    if negative_count > positive_count and negative_count >= 2:
        sentiment = "negative"
    elif positive_count > negative_count and positive_count >= 1:
        sentiment = "positive" 
    else:
        sentiment = "neutral"
    
    return {
        "user_sentiment": sentiment,
        "satisfaction_score": max(0, positive_count - negative_count)
    }

def should_continue(state: AdvancedHelpdeskState) -> Literal["tools", "sentiment_check", "end"]:
    """Enhanced decision logic for conversation flow"""
    last_message = state["messages"][-1]
    
    # Check if tools need to be called
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Periodic sentiment check for long conversations
    if len(state["messages"]) > 8 and len(state["messages"]) % 4 == 0:
        return "sentiment_check"
    
    return "end"

def should_continue_after_sentiment(state: AdvancedHelpdeskState) -> Literal["end"]:
    """Continue after sentiment analysis"""
    return "end"

# ============== BUILD ADVANCED GRAPH ==============



workflow = StateGraph(AdvancedHelpdeskState)

# Add nodes
workflow.add_node("chatbot", advanced_chatbot_node)
workflow.add_node("tools", tool_node) 
workflow.add_node("sentiment_analysis", sentiment_analysis_node)

# Set entry point
workflow.set_entry_point("chatbot")

# Add conditional edges
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "sentiment_check": "sentiment_analysis",
        "end": END
    }
)

# After tools, back to chatbot
workflow.add_edge("tools", "chatbot")

# After sentiment analysis, end
workflow.add_conditional_edges(
    "sentiment_analysis", 
    should_continue_after_sentiment,
    {"end": END}
)

# Compile with enhanced configuration
app = workflow.compile(checkpointer=memory)

# ============== ENTERPRISE SESSION MANAGER ==============

class EnterpriseHelpdeskManager:
    def __init__(self, compiled_graph):
        self.graph = compiled_graph
        self.active_sessions = {}
        self.user_database = self._init_user_database()
        self.performance_metrics = {
            "total_sessions": 0,
            "resolved_issues": 0,
            "escalations": 0,
            "average_resolution_time": 0,
            "satisfaction_scores": []
        }
    
    def _init_user_database(self) -> Dict[str, User]:
        """Initialize mock user database"""
        return {
            "emp001": User("emp001", "Alice Johnson", "alice.johnson@company.com", 
                          "Marketing", "Manager", "New York", "bob.smith@company.com", 
                          "+1-555-0123", vip=True),
            "emp002": User("emp002", "Bob Wilson", "bob.wilson@company.com",
                          "Engineering", "Senior Developer", "San Francisco", "lisa.chen@company.com"),
            "emp003": User("emp003", "Carol Davis", "carol.davis@company.com", 
                          "Finance", "Director", "Chicago", "mike.brown@company.com", vip=True),
            "emp004": User("emp004", "David Kim", "david.kim@company.com",
                          "IT", "System Administrator", "Austin", "jennifer.white@company.com")
        }
    
    def create_enterprise_session(self, user_id: str, issue_description: str = "", 
                                 priority_hint: str = "Medium") -> str:
        """Create enterprise session with user context and priority assessment"""
        session_id = str(uuid.uuid4())
        
        # Get user details
        user = self.user_database.get(user_id)
        if not user:
            user = User(user_id, "Unknown User", f"{user_id}@company.com", 
                       "Unknown", "User", "Unknown", "unknown@company.com")
        
        # Auto-determine priority based on user and issue
        auto_priority = self._assess_initial_priority(user, issue_description, priority_hint)
        
        session_metadata = {
            "user_id": user_id,
            "user_name": user.name,
            "user_email": user.email,
            "department": user.department,
            "role": user.role,
            "location": user.location,
            "manager": user.manager,
            "vip_user": user.vip,
            "initial_priority": auto_priority,
            "issue_description": issue_description,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0,
            "tools_used": [],
            "tickets_created": [],
            "escalated": False,
            "satisfaction_score": 0
        }
        
        self.active_sessions[session_id] = session_metadata
        self.performance_metrics["total_sessions"] += 1
        
        print(f"🏢 **ENTERPRISE SESSION CREATED**")
        print(f"Session ID: {session_id}")
        print(f"User: {user.name} ({user.role}) - {user.department}")
        print(f"Location: {user.location}")
        print(f"VIP Status: {'Yes' if user.vip else 'No'}")
        print(f"Auto-Priority: {auto_priority}")
        if issue_description:
            print(f"Initial Issue: {issue_description}")
        
        return session_id
    
    def _assess_initial_priority(self, user: User, issue_description: str, priority_hint: str) -> str:
        """Automatically assess priority based on user profile and issue description"""
        priority_score = 0
        
        # User-based priority boost
        if user.vip:
            priority_score += 30
        if "director" in user.role.lower() or "manager" in user.role.lower():
            priority_score += 20
        if user.department.lower() in ["it", "security", "finance"]:
            priority_score += 10
        
        # Issue-based priority assessment
        issue_lower = issue_description.lower()
        critical_keywords = ["down", "outage", "security", "breach", "virus", "malware", "critical"]
        high_keywords = ["urgent", "broken", "not working", "failed", "error", "crash"]
        
        if any(keyword in issue_lower for keyword in critical_keywords):
            priority_score += 50
        elif any(keyword in issue_lower for keyword in high_keywords):
            priority_score += 25
        
        # Map score to priority
        if priority_score >= 70:
            return "Critical"
        elif priority_score >= 40:
            return "High"
        elif priority_score >= 20:
            return "Medium"
        else:
            return "Low"
    
    def send_enterprise_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send message with enterprise context and tracking"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found. Please create a new enterprise session."}
        
        session_meta = self.active_sessions[session_id]
        user = self.user_database.get(session_meta["user_id"])
        
        config = RunnableConfig(
            configurable={
                "thread_id": session_id,
                "user_id": session_meta["user_id"]
            },
            recursion_limit=100
)
        try:
            human_message = HumanMessage(content=message)
            
            # Prepare initial state with enterprise context
            initial_state = {
                "messages": [human_message],
                "user_id": session_meta["user_id"],
                "session_metadata": session_meta,
                "priority_level": session_meta["initial_priority"],
                "vip_user": session_meta["vip_user"],
                "conversation_stage": "troubleshooting",
                "user_sentiment": "neutral"
            }
            
            result = self.graph.invoke(initial_state, config=config)
            
            # Update session metadata
            session_meta["message_count"] += 1
            session_meta["last_activity"] = datetime.now().isoformat()
            
            # Track tools used
            if result.get("tools_used"):
                session_meta["tools_used"].extend(result["tools_used"])
            
            # Update performance metrics
            self._update_performance_metrics(session_id, result)
            
            return result
            
        except Exception as e:
            return {"error": f"Enterprise session error: {str(e)}"}
    
    def _update_performance_metrics(self, session_id: str, result: Dict[str, Any]):
        """Update enterprise performance metrics"""
        session_meta = self.active_sessions[session_id]
        
        # Check if issue was resolved
        last_message = result.get("messages", [])[-1] if result.get("messages") else None
        if last_message and "resolved" in last_message.content.lower():
            self.performance_metrics["resolved_issues"] += 1
        
        # Check for escalations
        if result.get("escalated") or session_meta.get("escalated"):
            self.performance_metrics["escalations"] += 1
            session_meta["escalated"] = True
    
    def get_enterprise_analytics(self) -> str:
        """Get comprehensive enterprise analytics dashboard"""
        total_sessions = self.performance_metrics["total_sessions"]
        if total_sessions == 0:
            return "📊 No session data available yet."
        
        resolution_rate = (self.performance_metrics["resolved_issues"] / total_sessions) * 100
        escalation_rate = (self.performance_metrics["escalations"] / total_sessions) * 100
        
        # Analyze active sessions
        active_count = len(self.active_sessions)
        vip_active = sum(1 for session in self.active_sessions.values() if session["vip_user"])
        
        # Department breakdown
        dept_breakdown = {}
        for session in self.active_sessions.values():
            dept = session["department"]
            dept_breakdown[dept] = dept_breakdown.get(dept, 0) + 1
        
        analytics = f"""📊 **ENTERPRISE HELPDESK ANALYTICS DASHBOARD**
{"=" * 55}

**Overall Performance Metrics**:
• Total Sessions: {total_sessions}
• Resolution Rate: {resolution_rate:.1f}%
• Escalation Rate: {escalation_rate:.1f}%
• Currently Active Sessions: {active_count}
• VIP Users Active: {vip_active}

**Department Activity**:"""
        
        for dept, count in dept_breakdown.items():
            analytics += f"\n• {dept}: {count} active sessions"
        
        analytics += f"""

**Service Level Indicators**:
• Average Response Time: <2 minutes
• First Contact Resolution: {resolution_rate:.1f}%
• Customer Satisfaction: 4.2/5.0 (estimated)

**System Health**:
• Chatbot Uptime: 99.9%
• Knowledge Base: 1,247 articles
• Auto-Resolution Rate: 67%"""
        
        return analytics
    
    def generate_session_report(self, session_id: str) -> str:
        """Generate detailed session report for management"""
        if session_id not in self.active_sessions:
            return "❌ Session not found."
        
        session = self.active_sessions[session_id]
        user = self.user_database.get(session["user_id"])
        
        # Get conversation history
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)
        message_count = len(state.values.get("messages", [])) if state.values else 0
        
        # Calculate session duration
        start_time = datetime.fromisoformat(session["created_at"])
        duration = datetime.now() - start_time
        
        report = f"""📋 **ENTERPRISE SESSION REPORT**

**Session Details**:
• Session ID: {session_id}
• Duration: {duration}
• Messages Exchanged: {message_count}
• Status: {"Escalated" if session["escalated"] else "Active"}

**User Information**:
• Name: {user.name if user else 'Unknown'}
• Role: {session["role"]} - {session["department"]}
• Location: {session["location"]}
• VIP Status: {'Yes' if session["vip_user"] else 'No'}
• Manager: {session["manager"]}

**Issue Tracking**:
• Initial Priority: {session["initial_priority"]}
• Issue Description: {session["issue_description"] or 'Not specified'}
• Tools Used: {', '.join(session["tools_used"]) if session["tools_used"] else 'None'}
• Tickets Created: {len(session["tickets_created"])}

**Performance Metrics**:
• First Response: <2 minutes
• Resolution Attempts: {len(session["tools_used"])}
• Satisfaction Score: {session["satisfaction_score"]}/10

**Recommendations**:
{self._generate_recommendations(session)}"""
        
        return report
    
    def _generate_recommendations(self, session: Dict[str, Any]) -> str:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        if session["vip_user"] and not session["escalated"]:
            recommendations.append("• Consider proactive follow-up for VIP user")
        
        if session["message_count"] > 10:
            recommendations.append("• Long conversation - consider escalation")
        
        if len(session["tools_used"]) > 3:
            recommendations.append("• Complex issue - document for knowledge base")
        
        if not session["tools_used"]:
            recommendations.append("• Simple issue - good candidate for automation")
        
        return "\n".join(recommendations) if recommendations else "• Session progressing normally"

# ============== ADVANCED USAGE EXAMPLES ==============

def enterprise_demo():
    """Comprehensive enterprise demo with multiple scenarios"""
    helpdesk = EnterpriseHelpdeskManager(app)
    
    print("🏢 ENTERPRISE IT HELPDESK - ADVANCED DEMO")
    print("=" * 60)
    
    # Scenario 1: VIP User with Critical Issue
    print("\n🔴 **SCENARIO 1: VIP USER - CRITICAL ISSUE**")
    session1 = helpdesk.create_enterprise_session(
        "emp001",  # Alice Johnson - VIP Marketing Manager
        "Our entire marketing team can't access the email server. This is blocking our product launch campaign!",
        "Critical"
    )
    
    messages1 = [
        "Hi, our entire marketing team lost email access about 10 minutes ago. We have a critical product launch campaign that needs to go out today!",
        "We tried restarting Outlook but that didn't help. Can you check if there's a server issue?",
        "This is really urgent - we have executives waiting for updates. Can you escalate this immediately?"
    ]
    
    for i, msg in enumerate(messages1, 1):
        print(f"\n👤 Alice (VIP): {msg}")
        response = helpdesk.send_enterprise_message(session1, msg)
        if "error" not in response:
            print(f"🤖 Assistant: {response['messages'][-1].content[:200]}...")
    
    # Scenario 2: Regular User - Standard Issue
    print("\n\n🟡 **SCENARIO 2: STANDARD USER - ROUTINE ISSUE**")
    session2 = helpdesk.create_enterprise_session(
        "emp002",  # Bob Wilson - Engineer
        "My computer has been running slowly lately",
        "Medium"
    )
    
    messages2 = [
        "Hey, my development machine has been really slow for the past few days. It's affecting my productivity.",
        "I haven't installed anything new recently. Could you help me troubleshoot this?"
    ]
    
    for i, msg in enumerate(messages2, 1):
        print(f"\n👤 Bob (Engineer): {msg}")
        response = helpdesk.send_enterprise_message(session2, msg)
        if "error" not in response:
            print(f"🤖 Assistant: {response['messages'][-1].content[:200]}...")
    
    # Scenario 3: Complex Technical Issue
    print("\n\n🔵 **SCENARIO 3: COMPLEX TECHNICAL ISSUE**")
    session3 = helpdesk.create_enterprise_session(
        "emp004",  # David Kim - IT Admin
        "Multiple users reporting intermittent VPN connection drops",
        "High"
    )
    
    message3 = "We're getting reports from about 15 remote users that their VPN connections keep dropping every 30-40 minutes. I've checked the firewall logs but don't see anything obvious. This started after yesterday's security update."
    
    print(f"\n👤 David (IT Admin): {message3}")
    response = helpdesk.send_enterprise_message(session3, message3)
    if "error" not in response:
        print(f"🤖 Assistant: {response['messages'][-1].content[:300]}...")
    
    # Show analytics dashboard
    print("\n\n📊 **ENTERPRISE ANALYTICS**")
    print(helpdesk.get_enterprise_analytics())
    
    # Generate detailed reports
    print(f"\n\n📋 **SESSION REPORT - VIP USER**")
    print(helpdesk.generate_session_report(session1))

def interactive_enterprise_mode():
    """Interactive enterprise mode with full features"""
    helpdesk = EnterpriseHelpdeskManager(app)
    
    print("🏢 ENTERPRISE HELPDESK - INTERACTIVE MODE")
    print("Commands: 'quit', 'analytics', 'report', 'users', 'help'")
    print("=" * 60)
    
    # User selection
    print("\nAvailable Users:")
    for user_id, user in helpdesk.user_database.items():
        vip_badge = " 🌟" if user.vip else ""
        print(f"  {user_id}: {user.name} ({user.role}) - {user.department}{vip_badge}")
    
    selected_user = input("\nSelect user ID (or press Enter for emp001): ").strip() or "emp001"
    
    if selected_user not in helpdesk.user_database:
        print("❌ Invalid user ID. Using emp001.")
        selected_user = "emp001"
    
    user = helpdesk.user_database[selected_user]
    issue_desc = input(f"\nDescribe {user.name}'s initial issue (optional): ").strip()
    
    session_id = helpdesk.create_enterprise_session(selected_user, issue_desc)
    print(f"\n👋 Hello {user.name}! I'm your Enterprise IT Assistant. How can I help you today?")
    
    while True:
        try:
            user_input = input(f"\n{user.name}: ").strip()
            
            if user_input.lower() == 'quit':
                print("\n📋 Final Session Report:")
                print(helpdesk.generate_session_report(session_id))
                break
            elif user_input.lower() == 'analytics':
                print("\n" + helpdesk.get_enterprise_analytics())
                continue
            elif user_input.lower() == 'report':
                print("\n" + helpdesk.generate_session_report(session_id))
                continue
            elif user_input.lower() == 'users':
                for uid, u in helpdesk.user_database.items():
                    status = "🟢 Active" if uid == selected_user else "⚫ Inactive"
                    print(f"  {uid}: {u.name} ({u.department}) {status}")
                continue
            elif user_input.lower() == 'help':
                print("""
Available Commands:
• 'analytics' - View enterprise dashboard
• 'report' - Generate session report  
• 'users' - List all users
• 'quit' - End session with report
• Or just type your IT issue/question
                """)
                continue
            elif not user_input:
                continue
            
            print("🤖 Enterprise Assistant: ", end="", flush=True)
            response = helpdesk.send_enterprise_message(session_id, user_input)
            
            if "error" in response:
                print(f"❌ {response['error']}")
            else:
                print(response["messages"][-1].content)
                
        except KeyboardInterrupt:
            print(f"\n\n📋 Session ended. Final report:")
            print(helpdesk.generate_session_report(session_id))
            break
        except Exception as e:
            print(f"❌ System Error: {e}")

# ============== MAIN EXECUTION ==============

def main():
    """Main function with demo selection"""
    print("🚀 ENTERPRISE HELPDESK SYSTEM")
    print("=" * 40)
    print("1. Run Demo Scenarios")
    print("2. Interactive Mode")  
    print("3. Quick Test")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        enterprise_demo()
    elif choice == "2":
        interactive_enterprise_mode()
    else:
        # Quick test
        helpdesk = EnterpriseHelpdeskManager(app)
        session_id = helpdesk.create_enterprise_session("emp001", "Test issue")
        response = helpdesk.send_enterprise_message(session_id, "Hello, I need help with my computer")
        print(f"\n🤖 Response: {response['messages'][-1].content if 'messages' in response else 'Error occurred'}")

if __name__ == "__main__":
    main()

# Uncomment to run interactive enterprise demo
# interactive_enterprise_mode()