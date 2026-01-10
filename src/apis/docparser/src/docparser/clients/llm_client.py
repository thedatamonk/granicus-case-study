# chat_service/clients/llm_client.py

from abc import ABC, abstractmethod
from typing import Dict, Any
from openai import OpenAI
import json
from loguru import logger
from docparser.settings import get_settings

settings = get_settings()

SYSTEM_PROMPT = """
You are a data transformation engine.

You are given a JSON string representing tabular data, where:

* Each JSON object corresponds to one row from a CSV file.
* Keys are column headers.
* Values are cell values.

Your task is to **convert this JSON into a new structured JSON representation** according to the following rules:

### **Primary Objective**

Reorganize the data into a **hierarchical JSON structure** by:

1. **Detecting logical groupings** among rows when they exist.
2. **Grouping related rows under appropriate group identifiers** when possible.
3. **Otherwise, representing each row as its own independent section.**

### **How to Identify Groupings**

You must analyze the rows and determine whether:

* Some rows clearly belong to the same conceptual entity (e.g., same order_id, same user, same invoice, same product, same category, same document, etc.)
* Or whether each row is independent and unrelated.

Groupings may be inferred from:

* Repeated IDs
* Repeated names or keys
* Repeated entity references
* Any stable column or column combination that logically represents a parent entity

### **Output Structure Rules**

#### Case 1: **No Groupings Found**

* Output a JSON object where:

  * Each row becomes its own top-level section.
  * Each section must have a unique, human-readable identifier (e.g., `row_1`, `record_2`, or derived from the content if possible).
  * Each section contains all the fields from that row.

#### Case 2: **Groupings Found**

* Output a JSON object where:

  * Each top-level key represents one group (e.g., an order_id, customer_id, document_id, etc.).
  * The value for each group contains:

    * The shared/common fields (if applicable), and/or
    * A list or nested object containing the individual row-level entries belonging to that group.

### **Output Quality Requirements**

* Output **must be valid JSON**
* Output must be:

  * Deterministic
  * Consistent
  * Cleanly structured
  * Easy to read and logically organized
* Do **not** include explanations, comments, or markdown.
* Do **not** repeat redundant fields unnecessarily if they are clearly group-level attributes.
* Use **clear, semantically meaningful keys** for groups and nested structures.

### **Important Constraints**

* Do not hallucinate fields or values.
* Do not drop any information from the input.
* Do not invent groupings if the data does not justify them.
* Only group when there is a **clear, defensible structural pattern** in the data.

### **Decision Rule**

If you are unsure whether grouping is justified:
→ Default to **NO GROUPING** and treat each row as independent.

### **Final Output**

Return **only** the transformed JSON in the specified format ONLY.
No explanations. No commentary. No markdown.
"""

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
            
            user_prompt  = f"""
            Here is the JSON string\n\n
            {prompt}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Get the JSON
            result = json.loads(content)
            
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