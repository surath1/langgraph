
from dotenv import load_dotenv
load_dotenv()
import os
import logging

ENVIRONMENT=os.getenv("ENVIRONMENT", "development")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "o4-mini")

#https://home.openweathermap.org/api_keys
OPEN_WEATHER_MAP_API_KEY = os.getenv("OPEN_WEATHER_MAP_API_KEY", "")
#https://app.exchangerate-api.com/dashboard
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY", "")

## https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "H3942MV3QE476CDK")

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "180"))  # Default timeout in seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Default max retries
logger.info("------STARTING APPLICATION------")
logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Default Model Name: {DEFAULT_MODEL_NAME}")
logger.info(f"Health Check - http://0.0.0.0:8000/api/health")
logger.info("------Langgraph Multi Agent Application------")
