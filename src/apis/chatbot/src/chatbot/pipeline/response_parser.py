from typing import List, Dict, Any, Set
import re
from loguru import logger
from chatbot.serialisation import Source, ChatResponse

def extract_cited_sources(answer: str) -> Set[int]:
    """Extract source numbers from inline citations in answer."""
    # Match patterns like [Source 1], [Source 2], etc.
    pattern = r'\[Source (\d+)\]'
    matches = re.findall(pattern, answer)
    return set(int(n) for n in matches)

def validate_citations(
    sources_used_claimed: List[int],
    sources_cited_in_text: Set[int],
    available_sources: List[Dict[str, Any]]
) -> tuple[List[int], List[str]]:
    """Validate that claimed citations match actual citations and available sources."""
    warnings = []
    available_ids = {s["source_id"] for s in available_sources}
    
    # Check for invalid source IDs
    invalid_claimed = [sid for sid in sources_used_claimed if sid not in available_ids]
    if invalid_claimed:
        warnings.append(f"LLM claimed non-existent sources: {invalid_claimed}")
    
    invalid_cited = [sid for sid in sources_cited_in_text if sid not in available_ids]
    if invalid_cited:
        warnings.append(f"LLM cited non-existent sources in text: {invalid_cited}")
    
    # Check for mismatch between claimed and actual citations
    claimed_set = set(sources_used_claimed) & available_ids
    cited_set = sources_cited_in_text & available_ids
    
    if claimed_set != cited_set:
        warnings.append(
            f"Mismatch: claimed {claimed_set}, actually cited {cited_set}"
        )
    
    # Use union of both as valid sources
    valid_sources = list(claimed_set | cited_set)
    
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
    sources_used_claimed = llm_response.get("sources_used", [])
    confidence = llm_response.get("confidence", "medium").lower()
    
    # Validate confidence value
    if confidence not in ["high", "medium", "low"]:
        logger.warning(f"Invalid confidence '{confidence}', defaulting to 'medium'")
        confidence = "medium"
    
    # Extract citations from answer text
    sources_cited_in_text = extract_cited_sources(answer)
    
    # Validate citations
    valid_source_ids, warnings = validate_citations(
        sources_used_claimed,
        sources_cited_in_text,
        sources
    )
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Citation validation: {warning}")
    
    # Mark which sources were actually cited
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