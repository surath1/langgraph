
from Langgraph_multiagent_travel.backend.utils.llm_loader import ModelLoader
from prompts.prompt import SYSTEM_MESSAGE
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool
from backend.config.constant import logger

## Main class to build the graph
class GraphBuilder():
    """Class to build a workflow graph for a travel planning agent using LLMs and various tools."""
    def __init__(self,model_provider: str = "openai"):
        logger.info(f"Initializing GraphBuilder with model provider: {model_provider}")

        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        
        self.tools = []
        
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()
        
        self.tools.extend([* self.weather_tools.weather_tool_list, 
                           * self.place_search_tools.place_search_tool_list,
                           * self.calculator_tools.calculator_tool_list,
                           * self.currency_converter_tools.currency_converter_tool_list])
        
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        
        self.graph = None
        
        self.system_prompt = SYSTEM_MESSAGE
    
    # Main agent function
    def agent_function(self,state: MessagesState):
        """Function that processes the user input and generates a response using the LLM with tools."""
        
        logger.info("Agent function invoked.")
        user_question = state["messages"]
        input_question = [self.system_prompt] + user_question
        response = self.llm_with_tools.invoke(input_question)
        return {"messages": [response]}
    

    # Bild the graph with nodes and edges thi is the main function
    def build_graph(self):
        """Builds and returns the workflow graph."""

        logger.info("Building the workflow graph.")
        graph_builder=StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START,"agent")
        graph_builder.add_conditional_edges("agent",tools_condition)
        graph_builder.add_edge("tools","agent")
        graph_builder.add_edge("agent",END)
        self.graph = graph_builder.compile()
        return self.graph

    # Call method to build and return the graph    
    def __call__(self):
        """Calls the build_graph method and returns the constructed graph."""
        logger.info("Instance called.")
        return self.build_graph()