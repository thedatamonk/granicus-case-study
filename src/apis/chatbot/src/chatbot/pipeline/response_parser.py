from typing import List, Dict, Any
from loguru import logger
from chatbot.serialisation import Source, ChatResponse

def validate_citations(
    sources_cited: List[str],
    available_sources: List[Dict[str, Any]]
) -> tuple[set[str], List[str]]:
    """Validate that claimed citations match actual citations and available sources."""

    # Deduplicate cited sources
    sources_cited_unique = set(sources_cited)
    if len(sources_cited_unique) != len(sources_cited):
        logger.warning(f"LLM citing same source more than once.\n{sources_cited}")
    
    warnings = []
    
    available_source_ids = {s['source_id'] for s in available_sources}
    # Check for invalid source IDs
    non_existent_sources = [sid for sid in sources_cited if sid not in available_source_ids]
    if non_existent_sources:
        warnings.append(f"LLM claimed non-existent sources: {non_existent_sources}")
    
    # Check for mismatch between claimed and actual citations
    valid_sources = set(sources_cited) & available_source_ids
    
    return valid_sources, warnings

def parse_and_validate(
    llm_response: Dict[str, Any],
    sources: List[Dict[str, Any]],
    query: str,
    timing: Dict[str, int],
    model_name: str
) -> ChatResponse:
    """
    Parse LLM response and validate citations.
    """
    # Extract response fields
    answer = llm_response.get("answer", "")
    sources_cited = llm_response.get("sources_used", None)
    confidence = llm_response.get("confidence", None)
    if confidence:
        # Validate confidence value
        if confidence.lower() not in ["high", "medium", "low"]:
            logger.warning(f"Invalid confidence '{confidence}', defaulting to 'medium'")
            confidence = "medium"
    
    # Validate citations
    valid_source_ids = []
    warnings = []
    if sources_cited:
        logger.info("Validating citations...")
        valid_source_ids, warnings = validate_citations(sources_cited, sources)
    
        # Log warnings
        for warning in warnings:
            logger.warning(f"Citation validation: {warning}")
        
        # Mark valid sources as cited
        for source in sources:
            source["cited"] = source["source_id"] in valid_source_ids
        
    # Convert to Source models
    source_models = [Source(**s) for s in sources]
    
    # Build metadata
    metadata = {
        "model_used": model_name,
        "sources_retrieved": len(sources),
        "sources_cited": len(valid_source_ids),
        "citation_warnings": warnings,
        "latency_ms": timing
    }
    
    logger.info(
        f"Response parsed: {len(valid_source_ids)}/{len(sources)} sources cited, "
        f"confidence={confidence}"
    )
    
    return ChatResponse(
        query=query,
        answer=answer,
        sources=source_models,
        confidence=confidence,
        metadata=metadata
    )