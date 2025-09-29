"""
Helpdesk Chatbot - Simple CosmosDB Implementation
Assumes containers already exist, just simple read/write operations
"""

from azure.cosmos import CosmosClient
from datetime import datetime
from typing import Optional, Dict
import uuid


class HelpdeskChatbot:
    """Simple helpdesk chatbot with CosmosDB"""
    
    def __init__(self, cosmos_endpoint: str, cosmos_key: str, database_name: str = "helpdesk_db"):
        self.client = CosmosClient(cosmos_endpoint, cosmos_key)
        self.database = self.client.get_database_client(database_name)
        
        # Reference to containers (assumed to exist)
        self.users = self.database.get_container_client("users")
        self.conversations = self.database.get_container_client("conversations")
        self.messages = self.database.get_container_client("messages")
        self.tickets = self.database.get_container_client("tickets")
        self.checkpoints = self.database.get_container_client("checkpoints")
    
    # ==================== USER OPERATIONS ====================
    
    def get_or_create_user(self, email: str, name: str = None):
        """Get existing user or create new one"""
        # Try to find by email
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        items = list(self.users.query_items(query=query, max_item_count=1))
        
        if items:
            user = items[0]
            # Update last active
            user["last_active"] = datetime.utcnow().isoformat()
            self.users.upsert_item(user)
            return user["user_id"]
        
        # Create new user
        user_id = str(uuid.uuid4())
        user_doc = {
            "id": user_id,
            "user_id": user_id,
            "email": email,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "total_conversations": 0,
            "total_messages": 0
        }
        
        self.users.create_item(user_doc)
        return user_id
    
    def create_config(self, user_id: str):
        """Create LangGraph config for user"""
        return {
            "configurable": {
                "thread_id": user_id
            }
        }
    
    # ==================== CONVERSATION OPERATIONS ====================
    
    def start_conversation(self, user_id: str):
        """Start new conversation and return conversation_id"""
        conversation_id = str(uuid.uuid4())
        
        conv_doc = {
            "id": conversation_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "last_message_at": datetime.utcnow().isoformat(),
            "status": "active",
            "message_count": 0,
            "category": None,
            "intent": None,
            "resolved": False
        }
        
        self.conversations.create_item(conv_doc)
        return conversation_id
    
    def save_message(self, conversation_id: str, user_id: str, role: str, content: str):
        """Save message to conversation"""
        message_id = str(uuid.uuid4())
        
        message_doc = {
            "id": message_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,  # user or assistant
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.messages.create_item(message_doc)
        
        # Update conversation
        query = f"SELECT * FROM c WHERE c.conversation_id = '{conversation_id}'"
        convs = list(self.conversations.query_items(query=query, max_item_count=1))
        if convs:
            conv = convs[0]
            conv["message_count"] = conv.get("message_count", 0) + 1
            conv["last_message_at"] = datetime.utcnow().isoformat()
            self.conversations.upsert_item(conv)
        
        return message_id
    
    def update_conversation_metadata(self, conversation_id: str, category: str = None, 
                                    intent: str = None, resolved: bool = None):
        """Update conversation classification"""
        query = f"SELECT * FROM c WHERE c.conversation_id = '{conversation_id}'"
        convs = list(self.conversations.query_items(query=query, max_item_count=1))
        
        if convs:
            conv = convs[0]
            if category:
                conv["category"] = category
            if intent:
                conv["intent"] = intent
            if resolved is not None:
                conv["resolved"] = resolved
            
            self.conversations.upsert_item(conv)
    
    def get_conversation_history(self, conversation_id: str):
        """Get all messages from a conversation"""
        query = f"SELECT * FROM c WHERE c.conversation_id = '{conversation_id}' ORDER BY c.timestamp ASC"
        return list(self.messages.query_items(query=query))
    
    # ==================== TICKET OPERATIONS ====================
    
    def create_ticket(self, user_id: str, conversation_id: str, title: str, description: str):
        """Create support ticket"""
        ticket_id = str(uuid.uuid4())
        
        ticket_doc = {
            "id": ticket_id,
            "ticket_id": ticket_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "title": title,
            "description": description,
            "status": "open",
            "priority": "medium",
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.tickets.create_item(ticket_doc)
        
        # Update conversation
        self.update_conversation_metadata(conversation_id, resolved=False)
        
        return ticket_id
    
    # ==================== EXTRACTION OPERATIONS ====================
    
    def get_user_conversations(self, user_id: str):
        """Get all conversations for a user"""
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.started_at DESC"
        return list(self.conversations.query_items(query=query))
    
    def get_user_tickets(self, user_id: str):
        """Get all tickets for a user"""
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.created_at DESC"
        return list(self.tickets.query_items(query=query))
    
    def get_open_tickets(self):
        """Get all open tickets"""
        query = "SELECT * FROM c WHERE c.status = 'open' ORDER BY c.created_at DESC"
        return list(self.tickets.query_items(query=query))
    
    def get_daily_stats(self, date: str):
        """Get statistics for a specific date (YYYY-MM-DD)"""
        start = f"{date}T00:00:00Z"
        end = f"{date}T23:59:59Z"
        
        query = f"SELECT * FROM c WHERE c.started_at >= '{start}' AND c.started_at <= '{end}'"
        conversations = list(self.conversations.query_items(query=query))
        
        return {
            "date": date,
            "total_conversations": len(conversations),
            "resolved": len([c for c in conversations if c.get("resolved")]),
            "active_users": len(set(c["user_id"] for c in conversations))
        }


# ==================== CHECKPOINT SAVER (from previous example) ====================

from langgraph.checkpoint.base import BaseCheckpointSaver

class CosmosDBSaver(BaseCheckpointSaver):
    """Simple CosmosDB checkpoint saver"""
    
    def __init__(self, cosmos_endpoint: str, cosmos_key: str, database_name: str):
        self.client = CosmosClient(cosmos_endpoint, cosmos_key)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client("checkpoints")
    
    def put(self, config, checkpoint, metadata):
        """Save checkpoint"""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = checkpoint["id"]
        
        document = {
            "id": f"{thread_id}_{checkpoint_id}",
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.container.upsert_item(document)
        return config
    
    def get(self, config):
        """Get latest checkpoint"""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        query = f"SELECT * FROM c WHERE c.thread_id = '{thread_id}' ORDER BY c.timestamp DESC"
        items = list(self.container.query_items(query=query, max_item_count=1))
        
        if items:
            return items[0]["checkpoint"]
        return None
    
    def list(self, config):
        """List checkpoints"""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        query = f"SELECT * FROM c WHERE c.thread_id = '{thread_id}' ORDER BY c.timestamp DESC"
        items = list(self.container.query_items(query=query))
        
        return [(item["checkpoint"], item["metadata"]) for item in items]


# ==================== USAGE EXAMPLE ====================

def helpdesk_example():
    """Simple helpdesk usage example"""
    import os
    
    # Initialize
    helpdesk = HelpdeskChatbot(
        os.environ.get("COSMOS_ENDPOINT"),
        os.environ.get("COSMOS_KEY")
    )
    
    print("=== Helpdesk Chatbot Flow ===\n")
    
    # 1. User logs in
    user_id = helpdesk.get_or_create_user(
        email="john@company.com",
        name="John Doe"
    )
    print(f"✓ User ID: {user_id}")
    
    # 2. Create config for LangGraph
    config = helpdesk.create_config(user_id)
    print(f"✓ Config: {config}")
    
    # 3. Start conversation
    conversation_id = helpdesk.start_conversation(user_id)
    print(f"✓ Conversation ID: {conversation_id}\n")
    
    # 4. Save messages
    print("--- Chat Interaction ---")
    helpdesk.save_message(
        conversation_id, user_id, "user",
        "My laptop won't turn on"
    )
    print("User: My laptop won't turn on")
    
    helpdesk.save_message(
        conversation_id, user_id, "assistant",
        "I'll help you troubleshoot. Have you checked if it's plugged in?"
    )
    print("Bot: I'll help you troubleshoot. Have you checked if it's plugged in?")
    
    helpdesk.save_message(
        conversation_id, user_id, "user",
        "Yes, it's plugged in but still won't start"
    )
    print("User: Yes, it's plugged in but still won't start\n")
    
    # 5. Update conversation metadata
    helpdesk.update_conversation_metadata(
        conversation_id,
        category="hardware",
        intent="technical_issue"
    )
    print("✓ Updated conversation: category=hardware, intent=technical_issue")
    
    # 6. Create ticket if needed
    ticket_id = helpdesk.create_ticket(
        user_id, conversation_id,
        "Laptop won't power on",
        "User's laptop is not powering on despite being plugged in"
    )
    print(f"✓ Created ticket: {ticket_id}\n")
    
    # 7. Extract data
    print("--- Data Extraction ---")
    
    # Get conversation history
    messages = helpdesk.get_conversation_history(conversation_id)
    print(f"✓ Total messages: {len(messages)}")
    
    # Get user's conversations
    user_convs = helpdesk.get_user_conversations(user_id)
    print(f"✓ User conversations: {len(user_convs)}")
    
    # Get user's tickets
    user_tickets = helpdesk.get_user_tickets(user_id)
    print(f"✓ User tickets: {len(user_tickets)}")
    
    # Get daily stats
    today = datetime.utcnow().strftime("%Y-%m-%d")
    stats = helpdesk.get_daily_stats(today)
    print(f"✓ Today's stats: {stats}")


# ==================== API ENDPOINT EXAMPLE ====================

def api_chat_endpoint(user_email: str, message: str):
    """Example API endpoint for chat"""
    helpdesk = HelpdeskChatbot(
        os.environ.get("COSMOS_ENDPOINT"),
        os.environ.get("COSMOS_KEY")
    )
    
    # Get or create user
    user_id = helpdesk.get_or_create_user(user_email)
    
    # Get or create conversation (you might store conversation_id in session)
    # For this example, we'll start a new one
    conversation_id = helpdesk.start_conversation(user_id)
    
    # Save user message
    helpdesk.save_message(conversation_id, user_id, "user", message)
    
    # Here you would call your LangGraph chatbot
    # bot_response = app.invoke(...)
    bot_response = "I'm here to help!"  # Placeholder
    
    # Save bot message
    helpdesk.save_message(conversation_id, user_id, "assistant", bot_response)
    
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "response": bot_response
    }


if __name__ == "__main__":
    helpdesk_example()