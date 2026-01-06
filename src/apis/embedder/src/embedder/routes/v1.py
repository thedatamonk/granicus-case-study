from embedder.serialisation import EmbeddingRequest, EmbeddingResponse
from fastapi import APIRouter, Request
from sentence_transformers import SentenceTransformer

router = APIRouter(prefix="/v1")

@router.post("/embedding", response_model=EmbeddingResponse)
def embed(payload: EmbeddingRequest, request: Request) -> EmbeddingResponse:
    model: SentenceTransformer = request.app.state.model
    predictions = model.encode(payload.content)
    return EmbeddingResponse(embedding=predictions.tolist())