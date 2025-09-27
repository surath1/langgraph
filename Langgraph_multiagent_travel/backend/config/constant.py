
from dotenv import load_dotenv
load_dotenv()
import os


import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




API_TIMEOUT = int(os.getenv("API_TIMEOUT", "180"))  # Default timeout in seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Default max retries


