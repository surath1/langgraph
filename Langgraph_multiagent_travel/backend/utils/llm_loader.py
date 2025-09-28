from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config.constant import logger, OPENAI_API_KEY,GROQ_API_KEY,GOOGLE_API_KEY

""" Load configuration and provide model loading functionality. """
class ConfigLoader:
    def __init__(self):
        logger.info("Loading configuration...")
        self.config = load_config()
    
    def __getitem__(self, key):
        return self.config[key]

""" Default model name if not specified in config. TODO : Make it dynamic based on provider """


""" ModelLoader class to load LLM models based on provider. """
class ModelLoader(BaseModel):
    logger.info("Initializing Model Loader")
    model_provider: Literal["openai", "groq", "ollam", "gemini"] = "openai"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        logger.info("Loading LLM model...")
        logger.info(f"Loading model from provider: {self.model_provider}")
        if self.model_provider == "groq":

            logger.info("Loading LLM from Groq..............")
            groq_api_key = GROQ_API_KEY
            DEFAULT_MODEL_NAME = "" # Set a default model name if needed TODO
            model_name = self.config["llm"]["groq"]["models"].get("model_name", DEFAULT_MODEL_NAME)
            timeout = self.config["llm"]["groq"]["defaults"].get("timeout", 180)
            temperature = self.config["llm"]["groq"]["defaults"].get("temperature", 0.7)
            max_retries = self.config["llm"]["groq"]["defaults"].get("max_retries", 3)
            api_version = self.config["llm"]["groq"]["models"].get("api_version", None)            
            llm=ChatGroq(model=model_name, api_key=groq_api_key)

        elif self.model_provider == "openai":
            logger.info("Loading LLM from OpenAI..............")
            openai_api_key = OPENAI_API_KEY
            model_name = self.config["llm"]["openai"]["models"].get("model_name", DEFAULT_MODEL_NAME)
            timeout = self.config["llm"]["groq"]["defaults"].get("timeout", 180)
            temperature = self.config["llm"]["groq"]["defaults"].get("temperature", 0.7)
            max_retries = self.config["llm"]["groq"]["defaults"].get("max_retries", 3)
            api_version = self.config["llm"]["openai"]["models"].get("api_version", None)
            llm = ChatOpenAI(model_name=DEFAULT_MODEL_NAME, api_key=openai_api_key)

        elif self.model_provider == "ollama":
            logger.info("Loading LLM from Ollama..............")
            # Add Ollama model loading logic here
            raise NotImplementedError("Ollama model loading not implemented yet.")
        
        elif self.model_provider == "gemini":
            logger.info("Loading LLM from Gemini..............")
            gemini_api_key = GOOGLE_API_KEY
            # Add Gemini model loading logic here
            raise NotImplementedError("Gemini model loading not implemented yet.")    
        
        return llm
    