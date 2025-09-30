# LangGraph ReAct Agent with Google Gemini API
# Complete project implementation with multiple tools

import os
import json
import requests
import datetime
import logging
from typing import Literal
from pydantic import BaseModel, Field
# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
load_dotenv()

logging.getLogger("langchain").setLevel(logging.INFO)  # Suppress verbose LangChain logs
logging.basicConfig(level=logging.INFO)

# External libraries
from geopy.geocoders import Nominatim

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("No Gemini credentials found: set GOOGLE_API_KEY run gcloud auth application-default login")

class AgentState(MessagesState):
    """
    The state of the ReAct agent using MessagesState.
    
    MessagesState automatically provides:
    - messages: List of conversation messages with add_messages reducer
    
    Additional fields:
    - step_count: Counter for tracking agent steps  
    - user_location: Optional user location for context
    """
    step_count: int
    user_location: str | None

# ================================
# 1. TOOLS DEFINITION
# ================================

# Initialize geocoder for location-based tools
geolocator = Nominatim(user_agent="langgraph-react-agent")

class WeatherInput(BaseModel):
    location: str = Field(description="The city and state/country, e.g., 'San Francisco, CA' or 'Berlin, Germany'")
    date: str = Field(description="The date for weather forecast in YYYY-MM-DD format")

@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(location: str, date: str) -> dict:
    """
    Retrieves weather forecast for a given location and date using Open-Meteo API.
    Returns temperature data for each hour of the day.
    """
    try:
        # Geocode the location
        geo_location = geolocator.geocode(location)
        if not geo_location:
            return {"error": f"Location '{location}' not found"}
        
        # Get weather data
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": geo_location.latitude,
            "longitude": geo_location.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,weather_code",
            "start_date": date,
            "end_date": date,
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Format the response
        hourly_data = []
        for i, (time, temp, humidity, weather_code) in enumerate(zip(
            data["hourly"]["time"],
            data["hourly"]["temperature_2m"],
            data["hourly"]["relative_humidity_2m"],
            data["hourly"]["weather_code"]
        )):
            hourly_data.append({
                "time": time,
                "temperature_celsius": temp,
                "humidity_percent": humidity,
                "weather_code": weather_code
            })
        
        return {
            "location": location,
            "coordinates": {"lat": geo_location.latitude, "lon": geo_location.longitude},
            "date": date,
            "hourly_forecast": hourly_data[:12],  # First 12 hours for brevity
            "summary": f"Weather forecast for {location} on {date}"
        }
        
    except Exception as e:
        return {"error": f"Failed to get weather data: {str(e)}"}

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '2+2', '10*5-3', 'sqrt(16)'")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> dict:
    """
    Performs mathematical calculations. Supports basic arithmetic, sqrt, pow, etc.
    """
    try:
        # Safer evaluation using Python's AST (no direct eval of arbitrary code)
        import ast
        import math

        # Allowed functions and constants
        allowed_funcs = {
            'sqrt': math.sqrt,
            'pow': math.pow,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'round': round,
        }

        allowed_names = {
            'pi': math.pi,
            'e': math.e,
            'tau': getattr(math, 'tau', 2 * math.pi),
        }

        node = ast.parse(expression, mode='eval')

        def _eval(node):
            # Expression wrapper
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            # Binary operations
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = node.op
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.Pow):
                    return left ** right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                raise ValueError(f"Unsupported binary operator: {type(op).__name__}")

            # Unary operations
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.USub):
                    return -operand
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

            # Numbers / constants
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only int/float constants are allowed")

            # Function calls
            if isinstance(node, ast.Call):
                # support both 'sqrt(4)' and 'math.sqrt(4)'
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'math':
                    func_name = func.attr

                if func_name and func_name in allowed_funcs:
                    args = [_eval(a) for a in node.args]
                    return allowed_funcs[func_name](*args)

                raise ValueError(f"Function '{getattr(func, 'id', getattr(func, 'attr', str(func)))}' is not allowed")

            # Names (constants like pi)
            if isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"Use of name '{node.id}' is not allowed")

            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        result = _eval(node)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "error": f"Calculation error: {str(e)}",
            "success": False
        }

class SearchInput(BaseModel):
    query: str = Field(description="Search query to look up information")
    
@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> dict:
    """
    Performs a web search using DuckDuckGo API (simplified simulation).
    In a real implementation, you would use a proper search API.
    """
    # This is a simplified simulation - in practice you'd use a real search API
    return {
        "query": query,
        "results": [
            {
                "title": f"Search result for: {query}",
                "snippet": f"This is a simulated search result for the query '{query}'. In a real implementation, this would return actual web search results.",
                "url": "https://example.com"
            }
        ],
        "note": "This is a simulated search result. Implement with a real search API like SerpAPI, DuckDuckGo API, or Google Custom Search."
    }

class TimeInput(BaseModel):
    timezone: str = Field(description="Timezone (optional), e.g., 'UTC', 'US/Eastern', 'Europe/London'", default="UTC")

@tool("get_current_time", args_schema=TimeInput)
def get_current_time(timezone: str = "UTC") -> dict:
    """
    Gets the current date and time. Note: timezone parameter is accepted but simplified implementation uses local time.
    """
    try:
        if timezone == "UTC":
            current_time = datetime.datetime.now(datetime.timezone.utc)
        else:
            # For simplicity, return local time
            current_time = datetime.datetime.now()
            
        return {
            "current_time": current_time.isoformat(),
            "timezone": timezone if timezone == "UTC" else "Local",
            "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": current_time.timestamp()
        }
    except Exception as e:
        # Fallback to local time
        current_time = datetime.datetime.now()
        return {
            "current_time": current_time.isoformat(),
            "timezone": "Local",
            "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"Time error: {str(e)}"
        }

# List of all available tools
tools = [get_weather_forecast, calculator, web_search, get_current_time]

# ================================
# 2. MODEL INITIALIZATION
# ================================

def initialize_model():
    """Initialize the Gemini model with tools bound."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Latest model
            temperature=0.1,  # Lower temperature for more consistent responses
            max_retries=2,
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Bind tools to the model
        model_with_tools = llm.bind_tools(tools)
        return model_with_tools
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("💡 Make sure your GOOGLE_API_KEY is valid and you have internet connection")
        raise

# ================================
# 3. NODE DEFINITIONS
# ================================

def call_model(state: AgentState, config: RunnableConfig) -> dict:
    """
    Node that calls the language model with the current conversation.
    """
    model = initialize_model()
    
    # Add system message if this is the first interaction
    messages = state["messages"]
    if not messages or not any(isinstance(msg, AIMessage) for msg in messages):
        system_message = AIMessage(
            content="""You are a helpful ReAct (Reasoning and Acting) agent. You have access to several tools:

1. get_weather_forecast: Get weather information for any location and date
2. calculator: Perform mathematical calculations  
3. web_search: Search for information (simulated)
4. get_current_time: Get current time in any timezone

When helping users:
- Think step by step about what information you need
- Use tools when necessary to get accurate, up-to-date information
- Be conversational and helpful
- If you need to use multiple tools, explain your reasoning

Always be thorough in your responses and explain your thought process."""
        )
        messages = [system_message] + list(messages)
    
    response = model.invoke(messages, config)
    
    # Update step count
    current_step = state.get("step_count", 0)
    
    return {
        "messages": [response],
        "step_count": current_step + 1
    }

def call_tools(state: AgentState) -> dict:
    """
    Node that executes tool calls from the last AI message.
    """
    last_message = state["messages"][-1]
    tool_outputs = []
    
    # Create a mapping of tool names to tool functions
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Execute each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        try:
            if tool_name in tools_by_name:
                tool_result = tools_by_name[tool_name].invoke(tool_args)
                
                # Format the result as a string if it's a dict
                if isinstance(tool_result, dict):
                    tool_result = json.dumps(tool_result, indent=2)
                
                tool_outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
            else:
                tool_outputs.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found",
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
        except Exception as e:
            tool_outputs.append(
                ToolMessage(
                    content=f"Error executing tool '{tool_name}': {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
    
    return {"messages": tool_outputs}

# ================================
# 4. EDGE LOGIC
# ================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge that determines whether to continue with tool calls or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# ================================
# 5. GRAPH CONSTRUCTION
# ================================

def create_react_agent():
    """
    Creates and returns a compiled LangGraph ReAct agent.
    """
    # Initialize the graph with our state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tools)
    
    # Set the entry point
    workflow.set_entry_point("llm")
    
    # Add conditional edges from llm (use plural API)
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    
    # Add edge from tools back to llm
    workflow.add_edge("tools", "llm")
    
    # Compile the graph
    return workflow.compile()

# ================================
# 6. UTILITY FUNCTIONS
# ================================

def pretty_print_message(message: BaseMessage):
    """Pretty print a message with proper formatting."""
    if isinstance(message, HumanMessage):
        print(f"🧑 Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"🤖 Assistant: {message.content}")
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("   🔧 Tool calls:")
            for tool_call in message.tool_calls:
                print(f"      - {tool_call['name']}: {tool_call['args']}")
    elif isinstance(message, ToolMessage):
        print(f"🛠️ Tool ({message.name}): {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
    print()

def run_conversation(agent, initial_message: str, max_iterations: int = 10):
    """
    Runs a conversation with the agent.
    """
    # Initialize state
    state = {
        "messages": [HumanMessage(content=initial_message)],
        "step_count": 0,
        "user_location": None
    }
    
    print("🚀 Starting ReAct Agent Conversation")
    print("=" * 50)
    
    iteration = 0
    for current_state in agent.stream(state, stream_mode="values"):
        iteration += 1
        if iteration > max_iterations:
            print("⚠️  Maximum iterations reached")
            break
            
        last_message = current_state["messages"][-1]
        pretty_print_message(last_message)
        
        # Update state for next iteration
        state = current_state
    
    print("✅ Conversation completed")
    return state

# ================================
# 7. EXAMPLE USAGE AND TESTING
# ================================

def main():
    """
    Main function to demonstrate the ReAct agent.
    """
    print("🔧 Initializing LangGraph ReAct Agent with Google Gemini API...")
    
    # Create the agent
    agent = create_react_agent()
    
    print("✅ Agent initialized successfully!")
    print()
    
    # Test cases
    test_cases = [
        "What's the weather like in Tokyo today?",
        "Calculate the square root of 144 and then multiply it by 5", 
        "What time is it now?",
        f"What's the weather in London for {datetime.datetime.now().strftime('%Y-%m-%d')}?",
    ]
    
    print("🧪 Running test cases:")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}: {test_case}")
        print("-" * 60)
        
        try:
            final_state = run_conversation(agent, test_case)
            print(f"📊 Steps taken: {final_state.get('step_count', 0)}")
        except Exception as e:
            print(f"❌ Error in test case {i}: {str(e)}")
        
        print("\n" + "="*60 + "\n")

# ================================
# 8. INTERACTIVE MODE
# ================================

def interactive_mode():
    """
    Runs the agent in interactive mode for continuous conversation.
    """
    print("🎯 Starting Interactive ReAct Agent")
    print("Type 'quit' to exit, 'help' for available tools")
    print("-" * 50)
    
    agent = create_react_agent()
    
    # Initialize conversation state
    state = {
        "messages": [],
        "step_count": 0,
        "user_location": None
    }
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("🔧 Available tools:")
                for tool in tools:
                    print(f"   - {tool.name}: {tool.description}")
                continue
            elif not user_input:
                continue
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            # Run agent
            for current_state in agent.stream(state, stream_mode="values"):
                last_message = current_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    print(f"\n🤖 Assistant: {last_message.content}")
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        print("   🔧 Using tools...")
                
                # Update state
                state = current_state
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting the ReAct Agent main function")
    #main()
    

    # # Optionally run interactive mode
    run_interactive = input("\n🎯 Would you like to try interactive mode? (y/n): ").lower().strip()
    if run_interactive in ['y', 'yes']:
        interactive_mode()
    logging.info("Main function completed")