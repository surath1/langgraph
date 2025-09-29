"""
LangGraph with CosmosDB Checkpoint Example

This example demonstrates how to use Azure CosmosDB as a checkpoint
storage backend for LangGraph to persist conversation state.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from azure.cosmos import CosmosClient, PartitionKey
import os
import json
from datetime import datetime


# 1. Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int


# 2. Custom CosmosDB Checkpoint Saver
class CosmosDBSaver(BaseCheckpointSaver):
    """Simple CosmosDB checkpoint saver implementation"""
    
    def __init__(self, cosmos_endpoint: str, cosmos_key: str, database_name: str, container_name: str):
        self.client = CosmosClient(cosmos_endpoint, cosmos_key)
        self.database = self.client.create_database_if_not_exists(database_name)
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/thread_id")
        )
    
    def put(self, config, checkpoint, metadata):
        """Save a checkpoint to CosmosDB"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
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
        """Retrieve the latest checkpoint from CosmosDB"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        query = f"SELECT * FROM c WHERE c.thread_id = '{thread_id}' ORDER BY c.timestamp DESC"
        items = list(self.container.query_items(query=query, max_item_count=1))
        
        if items:
            return items[0]["checkpoint"]
        return None
    
    def list(self, config):
        """List all checkpoints for a thread"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        query = f"SELECT * FROM c WHERE c.thread_id = '{thread_id}' ORDER BY c.timestamp DESC"
        items = list(self.container.query_items(query=query))
        
        return [(item["checkpoint"], item["metadata"]) for item in items]


# 3. Define the graph nodes
def chatbot(state: State):
    """Simple chatbot node that counts messages"""
    return {
        "messages": [{"role": "assistant", "content": f"You've sent {state['count']} messages!"}],
        "count": state["count"] + 1
    }


# 4. Build the graph
def create_graph(checkpointer: BaseCheckpointSaver):
    """Create a LangGraph with checkpointing enabled"""
    
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("chatbot", chatbot)
    
    # Add edges
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


# 5. Usage Example
def main():
    # Initialize CosmosDB Checkpointer
    checkpointer = CosmosDBSaver(
        cosmos_endpoint=os.environ.get("COSMOS_ENDPOINT"),
        cosmos_key=os.environ.get("COSMOS_KEY"),
        database_name="langgraph_db",
        container_name="checkpoints"
    )
    
    # Create the graph
    app = create_graph(checkpointer)
    
    # Configuration with thread_id for session persistence
    config = {"configurable": {"thread_id": "user_123"}}
    
    # First conversation
    print("=== First Conversation ===")
    result1 = app.invoke(
        {"messages": [{"role": "user", "content": "Hello!"}], "count": 0},
        config=config
    )
    print(f"Response: {result1['messages'][-1]['content']}")
    print(f"Count: {result1['count']}\n")
    
    # Second conversation (state persisted)
    print("=== Second Conversation ===")
    result2 = app.invoke(
        {"messages": [{"role": "user", "content": "Hi again!"}]},
        config=config
    )
    print(f"Response: {result2['messages'][-1]['content']}")
    print(f"Count: {result2['count']}\n")
    
    # Different thread (new state)
    print("=== Different Thread ===")
    new_config = {"configurable": {"thread_id": "user_456"}}
    result3 = app.invoke(
        {"messages": [{"role": "user", "content": "Hello!"}], "count": 0},
        config=new_config
    )
    print(f"Response: {result3['messages'][-1]['content']}")
    print(f"Count: {result3['count']}")


if __name__ == "__main__":
    main()


# 6. Alternative: Using streaming with checkpoints
def streaming_example():
    """Example showing streaming with checkpoint persistence"""
    
    checkpointer = CosmosDBSaver(
        cosmos_endpoint=os.environ.get("COSMOS_ENDPOINT"),
        cosmos_key=os.environ.get("COSMOS_KEY"),
        database_name="langgraph_db",
        container_name="checkpoints"
    )
    
    app = create_graph(checkpointer)
    config = {"configurable": {"thread_id": "stream_user"}}
    
    # Stream the graph execution
    for chunk in app.stream(
        {"messages": [{"role": "user", "content": "Stream test"}], "count": 0},
        config=config
    ):
        print(chunk)

# if __name__ == "__main__":
#     streaming_example()        