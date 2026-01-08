from typing import List, Union

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    content: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]

class HeartbeatResult(BaseModel):
    healthy: bool
    healthy: bool
