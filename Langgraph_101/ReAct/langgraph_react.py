from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# # 1. Define tools
# wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import tool

# Initialize the Serper API wrapper.
# Ensure your SERPER_API_KEY environment variable is set.
serper_api_wrapper = GoogleSerperAPIWrapper()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
# Create a tool that uses the Serper API wrapper to run searches.
# The @tool decorator is a simple way to create a Tool from a function.
@tool
def serper_search(query: str) -> str:
    """Useful for when you need to search Google for general knowledge or real-time information."""
    return serper_api_wrapper.run(query)

# Example usage
print(serper_search.invoke("Who was the first president of the United States?"))

print("---"*10)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("No Gemini credentials found: set GOOGLE_API_KEY run gcloud auth application-default login")

# 2. Create the agent with a single line
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Latest model
    temperature=0.1,  # Lower temperature for more consistent responses
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

# 2. Create the agent with a single line
agent = create_react_agent(
    model=llm,
    tools=[serper_search],
    prompt="You are a helpful assistant. Use tools to answer questions."
)

# 3. Invoke the agent with a user query
result = agent.invoke({"messages": [{"role": "user", "content": "Who won the NBA Finals in 2024? and how much time they won"}]})
print(result['messages'][-1].content)
