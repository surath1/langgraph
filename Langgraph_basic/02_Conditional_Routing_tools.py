from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os   
from dotenv import load_dotenv
load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model="gpt-4o", api_key=open_ai_key)
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)
app = builder.compile()

print("--------------------------------")

#display(Image(app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))


messages = [HumanMessage(content="Hello AI, 10 multiplied by 20 plz ?")]
messages = app.invoke({"messages": messages})
#print(messages['messages'])
for m in messages['messages']:
    m.pretty_print()

