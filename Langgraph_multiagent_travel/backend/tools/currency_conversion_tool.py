import os
from utils.currency_converter import CurrencyConverter
from typing import List
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from backend.config.constant import logger

class CurrencyConverterTool:
    def __init__(self):
        
        logger.info("Initializing CurrencyConverterTool.")
        self.api_key = os.environ.get("EXCHANGE_RATE_API_KEY")
        self.currency_service = CurrencyConverter(self.api_key)
        self.currency_converter_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the currency converter tool"""
        @tool
        def convert_currency(amount:float, from_currency:str, to_currency:str):
            """Convert amount from one currency to another"""

            logger.info(f"Converting {amount} from {from_currency} to {to_currency}.")
            return self.currency_service.convert(amount, from_currency, to_currency)
        
        return [convert_currency]