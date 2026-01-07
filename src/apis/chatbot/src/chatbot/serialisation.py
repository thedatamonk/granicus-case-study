from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    """Incoming chat request."""
    query: str = Field(..., description="User's question")
    conversation_history: List[Message] = Field(
        default=[],
        description="Previous messages in conversation"
    )

class Source(BaseModel):
    """Single source/chunk used in response."""
    source_id: int = Field(..., description="Citation number [1], [2], etc.")
    chunk_text: str
    filename: str
    chunk_index: int
    doc_type: str
    relevance_score: float = Field(..., description="Similarity score (0-1, lower is better)")
    cited: bool = Field(default=False, description="Was this source cited in answer?")

class ChatResponse(BaseModel):
    """Response with grounded answer and citations."""
    query: str
    answer: str = Field(..., description="Answer with inline citations like [Source 1]")
    sources: List[Source]
    confidence: str = Field(..., description="high, medium, or low")
    metadata: dict = Field(
        default={},
        description="Latency, model info, etc."
    )