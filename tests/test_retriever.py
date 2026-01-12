import pytest
from unittest.mock import Mock, patch

from chatbot.pipeline.retriever import Retriever, get_retriever


# ============ Fixtures ============

@pytest.fixture
def mock_embedder():
    """Mock embedder client."""
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return embedder


@pytest.fixture
def mock_vecdb():
    """Mock Weaviate vector database client."""
    vecdb = Mock()
    vecdb.search.return_value = [
        {
            "source_id": "pdf-doc1-abc123",
            "chunk_text": "Machine learning is a subset of AI.",
            "doc_type": "pdf",
            "distance": 0.95
        },
        {
            "source_id": "csv-doc2-def456",
            "chunk_text": "Neural networks process data.",
            "doc_type": "csv",
            "distance": 0.87
        }
    ]
    return vecdb


@pytest.fixture
def mock_settings():
    """Mock settings with retrieval parameters."""
    settings = Mock()
    settings.max_sources = 10
    settings.similarity_threshold = 0.7
    return settings


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "What is machine learning?"


class TestRetriever:
    # ===== Initialization =====
    
    @patch('chatbot.pipeline.retriever.get_embedder_client')
    @patch('chatbot.pipeline.retriever.get_weaviate_client')
    def test_init_initializes_clients(
        self, 
        mock_get_weaviate, 
        mock_get_embedder
    ):
        """Test that Retriever initializes embedder and vector DB clients."""
        # Arrange
        mock_embedder = Mock()
        mock_vecdb = Mock()
        mock_get_embedder.return_value = mock_embedder
        mock_get_weaviate.return_value = mock_vecdb
        
        # Act
        retriever = Retriever()
        
        # Assert
        assert retriever.embedder == mock_embedder
        assert retriever.vecdb == mock_vecdb
        mock_get_embedder.assert_called_once()
        mock_get_weaviate.assert_called_once()
    
    
    # ===== Retrieve - Embedding Failure =====
    
    @patch('chatbot.pipeline.retriever.get_settings')
    @patch('chatbot.pipeline.retriever.get_embedder_client')
    @patch('chatbot.pipeline.retriever.get_weaviate_client')
    @patch('chatbot.pipeline.retriever.logger')
    def test_retrieve_raises_on_embedding_failure(
        self,
        mock_logger,
        mock_get_weaviate,
        mock_get_embedder,
        mock_get_settings,
        mock_vecdb,
        mock_settings,
        sample_query
    ):
        """Test that embedding failure propagates exception and logs error."""
        # Arrange
        mock_embedder_failing = Mock()
        mock_embedder_failing.embed_query.side_effect = Exception("Embedding service down")
        mock_get_embedder.return_value = mock_embedder_failing
        mock_get_weaviate.return_value = mock_vecdb
        mock_get_settings.return_value = mock_settings
        
        retriever = Retriever()
        
        # Act & Assert
        with pytest.raises(Exception, match="Embedding service down"):
            retriever.retrieve(sample_query)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "Embedding failed" in str(mock_logger.error.call_args)
        
        # Verify vector DB was never called
        mock_vecdb.search.assert_not_called()
    
    # ===== Retrieve - Retrieval Failure =====
    
    @patch('chatbot.pipeline.retriever.get_settings')
    @patch('chatbot.pipeline.retriever.get_embedder_client')
    @patch('chatbot.pipeline.retriever.get_weaviate_client')
    @patch('chatbot.pipeline.retriever.logger')
    def test_retrieve_raises_on_vecdb_failure(
        self,
        mock_logger,
        mock_get_weaviate,
        mock_get_embedder,
        mock_get_settings,
        mock_embedder,
        mock_settings,
        sample_query
    ):
        """Test that vector DB failure propagates exception after successful embedding."""
        # Arrange
        mock_vecdb_failing = Mock()
        mock_vecdb_failing.search.side_effect = Exception("Weaviate connection timeout")
        mock_get_embedder.return_value = mock_embedder
        mock_get_weaviate.return_value = mock_vecdb_failing
        mock_get_settings.return_value = mock_settings
        
        retriever = Retriever()
        
        # Act & Assert
        with pytest.raises(Exception, match="Weaviate connection timeout"):
            retriever.retrieve(sample_query)
        
        # Verify embedding succeeded
        mock_embedder.embed_query.assert_called_once_with(sample_query)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "Retrieval failed" in str(mock_logger.error.call_args)
    
    # ===== Get Retriever Singleton =====
    
    @patch('chatbot.pipeline.retriever.get_embedder_client')
    @patch('chatbot.pipeline.retriever.get_weaviate_client')
    def test_get_retriever_returns_singleton_instance(
        self,
        mock_get_weaviate,
        mock_get_embedder
    ):
        """Test that get_retriever returns same instance (singleton pattern)."""
        # Arrange
        mock_get_embedder.return_value = Mock()
        mock_get_weaviate.return_value = Mock()
        
        # Reset global state
        import chatbot.pipeline.retriever as retriever_module
        retriever_module._retriever = None
        
        # Act
        retriever1 = get_retriever()
        retriever2 = get_retriever()
        
        # Assert
        assert retriever1 is retriever2
        assert isinstance(retriever1, Retriever)
        
        # Verify clients initialized only once
        assert mock_get_embedder.call_count == 1
        assert mock_get_weaviate.call_count == 1