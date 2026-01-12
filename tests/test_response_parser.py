# tests/test_response_parser.py

import pytest
from unittest.mock import patch

from chatbot.pipeline.response_parser import (
    validate_citations,
    parse_and_validate
)
from chatbot.serialisation import ChatResponse


# ============ Fixtures ============

@pytest.fixture
def sample_sources():
    """Sample sources for testing."""
    return [
        {
            "source_id": "pdf-sample-doc1",
            "chunk_text": "Machine learning is a subset of AI.",
            "doc_type": "pdf",
            "relevance_score": 0.95,
            "cited": False
        },
        {
            "source_id": "md-sample-doc1",
            "chunk_text": "Neural networks mimic biological neurons.",
            "doc_type": "pdf",
            "relevance_score": 0.88,
            "cited": False
        },
        {
            "source_id": "csv-sample-doc1",
            "chunk_text": "Deep learning uses multiple layers.",
            "doc_type": "txt",
            "relevance_score": 0.82,
            "cited": False
        }
    ]


@pytest.fixture
def sample_timing():
    """Sample timing metrics."""
    return {
        "embedding_ms": 150,
        "retrieval_ms": 200,
        "llm_ms": 1500
    }


class TestResponseParser:
    # ===== Validate Citations =====
    
    def test_validate_perfect_match_no_warnings(self, sample_sources):
        """Test when claimed and cited sources match perfectly."""
        # Arrange
        sources_cited = ["pdf-sample-doc1"]
        
        # Act
        valid_sources, warnings = validate_citations(
            sources_cited,
            sample_sources
        )
        
        # Assert
        assert set(valid_sources) == {"pdf-sample-doc1"}
        assert warnings == []
    
    def test_validate_invalid_source_ids_filtered_out(self, sample_sources):
        """Test that non-existent source IDs are filtered and warned."""
        # Arrange
        sources_cited = ["pdf-sample-doc1", "html-sample-doc1"]  # html-sample-doc1 doesn't exist
        
        # Act
        valid_sources, warnings = validate_citations(
            sources_cited,
            sample_sources
        )
        
        # Assert
        assert "html-sample-doc1" not in valid_sources
        assert len(warnings) == 1  # Warnings must be generated for all non-existen sources, in this case # of warnings should be 1
    
    # ===== Parse and Validate =====
    
    def test_parse_valid_response_returns_chat_response(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test parsing a valid LLM response returns ChatResponse."""
        # Arrange
        llm_response = {
            "answer": "ML is AI. Neural nets exist.",
            "sources_used": ["pdf-sample-doc1"],
            "confidence": "high"
        }
        query = "What is ML?"
        model_name = "gpt-4o"
        
        # Act
        with patch('chatbot.pipeline.response_parser.logger'):
            result = parse_and_validate(
                llm_response,
                sample_sources,
                query,
                sample_timing,
                model_name
            )
        
        # Assert
        assert isinstance(result, ChatResponse)
        assert result.query == query
        assert result.confidence == "high"
        assert len(result.sources) == 3

    def test_parse_includes_correct_metadata(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test that metadata includes all required fields."""
        # Arrange
        llm_response = {
            "answer": "ML is AI. Neural nets exist.",
            "sources_used": ["pdf-sample-doc1"],
            "confidence": "high"
        }
        query = "What is ML?"
        model_name = "gpt-4o"
        
        # Act
        with patch('chatbot.pipeline.response_parser.logger'):
            result = parse_and_validate(
                llm_response,
                sample_sources,
                query,
                sample_timing,
                model_name
            )
        
        # Assert
        assert result.metadata["model_used"] == model_name
        assert result.metadata["sources_retrieved"] == 3
        assert result.metadata["sources_cited"] == 1
        assert result.metadata["latency_ms"] == sample_timing
        assert "citation_warnings" in result.metadata