from pydantic import BaseModel
from typing import Union, List


class EmbeddingRequest(BaseModel):
    content: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]