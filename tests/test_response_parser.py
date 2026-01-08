# tests/test_response_parser.py

import pytest
from unittest.mock import patch

from chatbot.pipeline.response_parser import (
    extract_cited_sources,
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
            "source_id": 1,
            "chunk_text": "Machine learning is a subset of AI.",
            "filename": "ml_intro.pdf",
            "chunk_index": 0,
            "doc_type": "pdf",
            "relevance_score": 0.95,
            "cited": False
        },
        {
            "source_id": 2,
            "chunk_text": "Neural networks mimic biological neurons.",
            "filename": "neural_nets.pdf",
            "chunk_index": 1,
            "doc_type": "pdf",
            "relevance_score": 0.88,
            "cited": False
        },
        {
            "source_id": 3,
            "chunk_text": "Deep learning uses multiple layers.",
            "filename": "deep_learning.txt",
            "chunk_index": 0,
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
    # ===== Extract Citations =====
    
    def test_extract_multiple_citations(self):
        """Test extracting multiple citations from text."""
        # Arrange
        answer = "First fact [Source 1]. Second fact [Source 2]. Third [Source 3]."
        
        # Act
        cited = extract_cited_sources(answer)
        
        # Assert
        assert cited == {1, 2, 3}
    
    def test_extract_no_citations_returns_empty_set(self):
        """Test that text without citations returns empty set."""
        # Arrange
        answer = "This text has no citations at all."
        
        # Act
        cited = extract_cited_sources(answer)
        
        # Assert
        assert cited == set()
    
    # ===== Validate Citations =====
    
    def test_validate_perfect_match_no_warnings(self, sample_sources):
        """Test when claimed and cited sources match perfectly."""
        # Arrange
        sources_used_claimed = [1, 2]
        sources_cited_in_text = {1, 2}
        
        # Act
        valid_sources, warnings = validate_citations(
            sources_used_claimed,
            sources_cited_in_text,
            sample_sources
        )
        
        # Assert
        assert set(valid_sources) == {1, 2}
        assert warnings == []
    
    def test_validate_mismatch_generates_warning(self, sample_sources):
        """Test that mismatched claimed vs cited generates warning."""
        # Arrange
        sources_used_claimed = [1, 2]
        sources_cited_in_text = {1}  # Only cited 1
        
        # Act
        valid_sources, warnings = validate_citations(
            sources_used_claimed,
            sources_cited_in_text,
            sample_sources
        )
        
        # Assert
        assert set(valid_sources) == {1, 2}  # Union
        assert len(warnings) > 0
        assert "Mismatch" in warnings[0]
    
    def test_validate_invalid_source_ids_filtered_out(self, sample_sources):
        """Test that non-existent source IDs are filtered and warned."""
        # Arrange
        sources_used_claimed = [1, 99]  # 99 doesn't exist
        sources_cited_in_text = {1, 88}  # 88 doesn't exist
        
        # Act
        valid_sources, warnings = validate_citations(
            sources_used_claimed,
            sources_cited_in_text,
            sample_sources
        )
        
        # Assert
        assert 99 not in valid_sources
        assert 88 not in valid_sources
        assert len(warnings) >= 2  # Warnings for both invalid IDs
    
    # ===== Parse and Validate =====
    
    def test_parse_valid_response_returns_chat_response(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test parsing a valid LLM response returns ChatResponse."""
        # Arrange
        llm_response = {
            "answer": "ML is AI [Source 1]. Neural nets exist [Source 2].",
            "sources_used": [1, 2],
            "confidence": "high"
        }
        query = "What is ML?"
        model_name = "claude-sonnet-4"
        
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
    
    def test_parse_marks_cited_sources_correctly(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test that only cited sources are marked as cited=True."""
        # Arrange
        llm_response = {
            "answer": "Only using [Source 1] here.",
            "sources_used": [1],
            "confidence": "medium"
        }
        query = "Test"
        
        # Act
        with patch('chatbot.pipeline.response_parser.logger'):
            result = parse_and_validate(
                llm_response,
                sample_sources,
                query,
                sample_timing,
                "test-model"
            )
        
        # Assert
        assert result.sources[0].cited is True   # Source 1
        assert result.sources[1].cited is False  # Source 2
        assert result.sources[2].cited is False  # Source 3
    
    def test_parse_handles_missing_fields_with_defaults(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test that missing fields use default values."""
        # Arrange
        llm_response = {}  # All fields missing
        query = "Test"
        
        # Act
        with patch('chatbot.pipeline.response_parser.logger'):
            result = parse_and_validate(
                llm_response,
                sample_sources,
                query,
                sample_timing,
                "test-model"
            )
        
        # Assert
        assert result.answer == ""  # Default
        assert result.confidence == "medium"  # Default
    
    def test_parse_invalid_confidence_defaults_to_medium(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test that invalid confidence values default to 'medium'."""
        # Arrange
        llm_response = {
            "answer": "Test",
            "sources_used": [],
            "confidence": "super_high"  # Invalid
        }
        query = "Test"
        
        # Act
        with patch('chatbot.pipeline.response_parser.logger') as mock_logger:
            result = parse_and_validate(
                llm_response,
                sample_sources,
                query,
                sample_timing,
                "test-model"
            )
        
        # Assert
        assert result.confidence == "medium"
        mock_logger.warning.assert_called()  # Warning logged
    
    def test_parse_includes_correct_metadata(
        self, 
        sample_sources, 
        sample_timing
    ):
        """Test that metadata includes all required fields."""
        # Arrange
        llm_response = {
            "answer": "Test [Source 1].",
            "sources_used": [1],
            "confidence": "high"
        }
        query = "Test"
        model_name = "claude-sonnet-4"
        
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