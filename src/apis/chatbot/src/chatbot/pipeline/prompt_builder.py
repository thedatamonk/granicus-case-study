from typing import List, Dict, Any
from chatbot.serialisation import Message

SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions based ONLY on the provided context sources.

CRITICAL RULES:
1. Answer ONLY using information from the provided [Source N] contexts below
2. Cite sources inline using the format [Source N] where N is the source number
3. If the answer requires information not in the sources, respond with "I don't have enough information to answer that question."
4. Do not use any external knowledge or make assumptions beyond what's in the sources
5. Be specific and cite the exact sources for each claim

RESPONSE FORMAT:
You must respond with valid JSON in this exact structure:
{
  "answer": "Your complete answer with inline citations like [Source 1] and [Source 2]",
  "sources_used": [1, 2],
  "confidence": "high"
}

Where:
- answer: Your response with inline [Source N] citations
- sources_used: Array of source numbers you cited (e.g., [1, 3, 5])
- confidence: Either "high", "medium", or "low" based on how well the sources answer the question
"""

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources into numbered context blocks."""
    if not sources:
        return "No context available."
    
    context_parts = []
    for source in sources:
        context_parts.append(
            f"[Source {source['source_id']}] "
            f"(from {source['filename']}, chunk {source['chunk_index']})\n"
            f"{source['chunk_text']}\n"
        )
    
    return "\n".join(context_parts)

def format_conversation_history(history: List[Message]) -> str:
    """Format conversation history for context."""
    if not history:
        return ""
    
    history_parts = ["Previous Conversation:"]
    for msg in history:
        role = msg.role.capitalize()
        history_parts.append(f"{role}: {msg.content}")
    
    return "\n".join(history_parts) + "\n\n"

def build_prompt(
    query: str,
    sources: List[Dict[str, Any]],
    conversation_history: List[Message] = None
) -> str:
    """Build complete prompt for LLM."""
    # Add system instruction
    prompt = SYSTEM_INSTRUCTION + "\n\n"
    
    # Add context sources
    prompt += "CONTEXT SOURCES:\n"
    prompt += format_sources(sources)
    prompt += "\n"
    
    # Add conversation history if provided
    if conversation_history:
        prompt += format_conversation_history(conversation_history)
    
    # Add current query
    prompt += f"USER QUESTION:\n{query}\n\n"
    prompt += "Provide your JSON response now:"
    
    return prompt