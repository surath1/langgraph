import requests
from config.constant import logger

"""https://home.openweathermap.org/api_keys"""
""" A simple weather forecast tool using OpenWeatherMap API. """
class WeatherForecastTool:
    def __init__(self, api_key:str):
        logger.info("Weather Forecast Tool initialized with provided API key.")
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(self, place:str):
        """Get current weather of a place"""
        logger.info(f"Fetching current weather for place: {place}")
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": place,
                "appid": self.api_key,
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            raise e
    
    def get_forecast_weather(self, place:str):
        """Get weather forecast of a place"""
        logger.info(f"Fetching weather forecast for place: {place}")
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": place,
                "appid": self.api_key,
                "cnt": 10,
                "units": "metric"
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            raise e