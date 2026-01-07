from fastapi import APIRouter, HTTPException
from loguru import logger
import time
from chatbot.serialisation import ChatRequest, ChatResponse
from chatbot.settings import get_settings
from chatbot.pipeline.retriever import get_retriever
from chatbot.pipeline.prompt_builder import build_prompt
from chatbot.clients.llm_client import get_llm_client
from chatbot.pipeline.response_parser import parse_and_validate

settings = get_settings()

router = APIRouter(prefix="/v1")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start_time = time.time()
    logger.info("Processing query...")

    try:
        logger.info("Retrieving sources...")
        retriever = get_retriever()
        retrieval_start = time.time()
        sources = retriever.retrieve(
            query=request.query,
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        logger.debug(f"Retriever finished in {retrieval_time}ms")

        # Handle no results case
        if not sources:
            logger.warning("No relevant sources found")
            total_time = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                query=request.query,
                answer="I don't have relevant information to answer this question. Please try rephrasing or ask about a different topic.",
                sources=[],
                confidence="low",
                metadata={
                    "model_used": settings.llm_model,
                    "sources_retrieved": 0,
                    "sources_cited": 0,
                    "citation_warnings": [],
                    "latency_ms": {
                        "retrieval_ms": retrieval_time,
                        "llm_generation_ms": 0,
                        "total_ms": total_time
                    }
                }
            )
        
        logger.info("Building prompt...")
        prompt_start = time.time()
        
        prompt = build_prompt(
            query=request.query,
            sources=sources,
            conversation_history=request.conversation_history
        )

        prompt_time = int((time.time() - prompt_start) * 1000)
        logger.debug(f"Prompt built in {prompt_time}ms ({len(prompt)} chars)")

        logger.info("Generating response...")
        llm_start = time.time()
        llm_client = get_llm_client()
        llm_response = llm_client.generate(prompt)
        llm_time = int((time.time() - llm_start) * 1000)
        logger.info(f"LLM responded in {llm_time}ms")

        logger.info("Validating LLM response...")
        
        timing = {
            "retrieval_ms": retrieval_time,
            "llm_generation_ms": llm_time,
            "total_ms": int((time.time() - start_time) * 1000)
        }
        
        parse_start = time.time()
        response = parse_and_validate(
            llm_response=llm_response,
            sources=sources,
            timing=timing,
            query=request.query,
            model_name=settings.llm_model
        )
        parse_time = int((time.time() - parse_start) * 1000)
        logger.debug(f"Response parsed in {parse_time}ms")

        logger.info(
            f"Query completed: {timing['total_ms']}ms | "
            f"Sources: {len(sources)} | "
            f"Cited: {response.metadata['sources_cited']} | "
            f"Confidence: {response.confidence}"
        )

        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )