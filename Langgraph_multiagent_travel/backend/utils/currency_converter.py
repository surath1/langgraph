import requests
from backend.config.constant import logger

""" A simple currency converter class using ExchangeRate-API. """
class CurrencyConverter:
    def __init__(self, api_key: str):
        self.base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/"
        logger.info("Currency Converter initialized with provided API key.")
    
    def convert(self, amount:float, from_currency:str, to_currency:str):
        """Convert the amount from one currency to another"""

        logger.info(f"Converting {amount} from {from_currency} to {to_currency}")
        
        url = f"{self.base_url}/{from_currency}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("API call failed:", response.json())
        rates = response.json()["conversion_rates"]
        if to_currency not in rates:
            raise ValueError(f"{to_currency} not found in exchange rates.")
        return amount * rates[to_currency]