import yaml
from config.constant import logger

""" Load configuration from a YAML file. """
def load_config(config_path: str = "config/config.yaml") -> dict:
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
        logger.info(f"Config: {config}")

    return config
