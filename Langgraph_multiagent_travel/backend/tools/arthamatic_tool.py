import os
from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from backend.config.constant import logger

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    logger.info(f"Adding {a} and {b}")
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtract two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The difference of a and b.
    """
    logger.info(f"Subtracting {b} from {a}")
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    logger.info(f"Multiplying {a} and {b}")
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        float: The quotient of a and b.
    """
    logger.info(f"Dividing {a} by {b}")
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def currency_converter(from_curr: str, to_curr: str, value: float)->float:
    logger.info(f"Converting {value} from {from_curr} to {to_curr}")
    os.environ["ALPHAVANTAGE_API_KEY"] = os.getenv('ALPHAVANTAGE_API_KEY')
    alpha_vantage = AlphaVantageAPIWrapper()
    response = alpha_vantage._get_exchange_rate(from_curr, to_curr)
    exchange_rate = response['Realtime Currency Exchange Rate']['5. Exchange Rate']
    logger.info(f"Exchange rate from {from_curr} to {to_curr} is {exchange_rate}")
    
    return value * float(exchange_rate)