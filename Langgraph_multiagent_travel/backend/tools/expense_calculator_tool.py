from Langgraph_multiagent_travel.backend.utils.calculator import Calculator
from typing import List
from langchain.tools import tool
from backend.config.constant import logger

class CalculatorTool:
    def __init__(self):
        logger.info("Initializing CalculatorTool.")
        self.calculator = Calculator()
        self.calculator_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the calculator tool"""
        @tool
        def estimate_total_hotel_cost(price_per_night:str, total_days:float) -> float:
            """Calculate total hotel cost"""
            logger.info(f"Estimating total hotel cost for {total_days} days at {price_per_night} per night.")
            return self.calculator.multiply(price_per_night, total_days)
        
        @tool
        def calculate_total_expense(*costs: float) -> float:
            """Calculate total expense of the trip"""
            logger.info(f"Calculating total expense from costs: {costs}.")
            return self.calculator.calculate_total(*costs)
        
        @tool
        def calculate_daily_expense_budget(total_cost: float, days: int) -> float:
            """Calculate daily expense"""
            logger.info(f"Calculating daily expense budget for total cost {total_cost} over {days} days.")
            return self.calculator.calculate_daily_budget(total_cost, days)
        
        return [estimate_total_hotel_cost, calculate_total_expense, calculate_daily_expense_budget]