from utils.currency_converter import CurrencyConverter
from typing import List
from langchain.tools import tool
from config.constant import logger, EXCHANGE_RATE_API_KEY

"""Currency Conversion Tool using CurrencyConverter service"""
class CurrencyConversionTool:
    def __init__(self):

        logger.info("Initializing Currency Conversion .")
        """ https://app.exchangerate-api.com/dashboard """
        self.api_key = EXCHANGE_RATE_API_KEY
        self.currency_service = CurrencyConverter(self.api_key)
        self.currency_converter_tool_list = self._setup_tools()

    """ Setup all tools for the currency converter tool """
    def _setup_tools(self) -> List:
        logger.info("Setting up currency conversion tools.")
        @tool
        def convert_currency(amount:float, from_currency:str, to_currency:str):
            """Convert amount from one currency to another"""

            logger.info(f"Converting {amount} from {from_currency} to {to_currency}.")
            return self.currency_service.convert(amount, from_currency, to_currency)
        
        return [convert_currency]