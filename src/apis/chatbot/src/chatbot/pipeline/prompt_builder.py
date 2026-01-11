from typing import List, Dict, Any
from chatbot.serialisation import Message

SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions based ONLY on the provided context sources.

CRITICAL RULES:
1. Answer ONLY using information from the below context section.
2. The context is a list of facts taken from verified sources.
3. After answering the query, you MUST cite the sources that helped you answer the query. Make sure each source is cited only once in an answer.
4. If the answer requires information not in the sources, respond with "I don't have enough information to answer that question."
5. Do not use any external knowledge or make assumptions beyond what's in the sources
6. Keep your answers very concise, and to the point.

RESPONSE FORMAT:
You must respond with valid JSON in this exact structure:
{
  "answer": "Your complete answer",
  "sources_used": [source_id_1, source_id_2, ....],
  "confidence": "high"
}

Where:
- answer: Your response
- sources_used: List of source ids you cited
- confidence: Either "high", "medium", or "low" based on how well the sources answer the question
"""

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources into numbered context blocks."""
    if not sources:
        return "No context available."
    
    
    context_parts = []
    for source in sources:
        context_parts.append(
            f"[source_id: {source['source_id']}]\n"
            f"[source_content_start]\n{source['chunk_text']}\n[source_content_end]\n"
        )

    ctx_strings = "\n".join(context_parts)

    context = f"""
**START OF CONTEXT SECTION**

{ctx_strings}

**END OF CONTEXT SECTION**
"""
    
    return context

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