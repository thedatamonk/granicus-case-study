# chat_service/clients/llm_client.py

from abc import ABC, abstractmethod
from typing import Dict, Any
from openai import OpenAI
import json
from loguru import logger
from chatbot.settings import get_settings

settings = get_settings()

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Dict with keys: answer, sources_used, confidence
        """
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.llm_api_key)
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout
        self.temperature = settings.llm_temperature
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API with structured JSON output.
        
        Args:
            prompt: Full prompt with system instructions and context
            
        Returns:
            Parsed JSON response
            
        Raises:
            Exception: If API call fails or response is invalid
        """
        try:
            logger.debug(f"Calling OpenAI {self.model} with temperature={self.temperature}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based only on provided context. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            result = json.loads(content)
            
            # Validate required fields
            if "answer" not in result:
                raise ValueError("Response missing 'answer' field")
            
            # Set defaults for optional fields
            result.setdefault("sources_used", [])
            result.setdefault("confidence", "medium")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"LLM generation failed: {str(e)}")

def create_llm_client() -> BaseLLMClient:
    """Factory to create appropriate LLM client based on settings."""
    provider = settings.llm_provider.lower()
    
    if provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Singleton
_llm_client = None

def get_llm_client() -> BaseLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client()
    return _llm_client