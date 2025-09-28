from utils.weather_info import WeatherForecastTool
from langchain.tools import tool
from typing import List
from config.constant import logger, OPEN_WEATHER_MAP_API_KEY

"""Class to setup weather information tools using OpenWeatherMap API."""
class WeatherInfoTool:
    def __init__(self):
        
        logger.info("Initializing Weather Info Tool")
        """https://home.openweathermap.org/api_keys"""
        self.api_key = OPEN_WEATHER_MAP_API_KEY
        self.weather_service = WeatherForecastTool(self.api_key)
        self.weather_tool_list = self._setup_tools()
    
    """Setup weather information tools."""
    def _setup_tools(self) -> List:
        logger.info("Setting up weather tools")
        @tool
        def get_current_weather(city: str) -> str:
            """Get current weather for a city"""
            logger.info(f"Fetching current weather for city: {city}")
            weather_data = self.weather_service.get_current_weather(city)
            if weather_data:
                temp = weather_data.get('main', {}).get('temp', 'N/A')
                desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
                return f"Current weather in {city}: {temp}°C, {desc}"
            return f"Could not fetch weather for {city}"
        
        @tool
        def get_weather_forecast(city: str) -> str:
            """Get weather forecast for a city"""
            logger.info(f"Fetching weather forecast for city: {city}")
            forecast_data = self.weather_service.get_forecast_weather(city)
            if forecast_data and 'list' in forecast_data:
                forecast_summary = []
                for i in range(len(forecast_data['list'])):
                    item = forecast_data['list'][i]
                    date = item['dt_txt'].split(' ')[0]
                    temp = item['main']['temp']
                    desc = item['weather'][0]['description']
                    forecast_summary.append(f"{date}: {temp} degree celcius , {desc}")
                return f"Weather forecast for {city}:\n" + "\n".join(forecast_summary)
            return f"Could not fetch forecast for {city}"
    
        return [get_current_weather, get_weather_forecast]