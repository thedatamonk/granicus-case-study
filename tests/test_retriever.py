import pytest
from unittest.mock import Mock, patch

from chatbot.pipeline.retriever import Retriever, get_retriever


# ============ Fixtures ============

@pytest.fixture
def mock_settings():
    """Mock settings with default values."""
    with patch('chatbot.settings.get_settings') as mock:
        settings = Mock()
        settings.max_sources = 5
        settings.similarity_threshold = 0.7
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_embedder():
    """Mock embedding client."""
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Fixed 5D vector
    return embedder


@pytest.fixture
def mock_weaviate():
    """Mock Weaviate client."""
    client = Mock()
    return client


@pytest.fixture
def sample_vector_db_results():
    """Sample results from vector database."""
    return [
        {
            "chunk_text": "Machine learning is a subset of artificial intelligence.",
            "filename": "ml_intro.pdf",
            "chunk_index": 0,
            "doc_type": "pdf",
            "distance": 0.95
        },
        {
            "chunk_text": "Neural networks are inspired by biological neurons.",
            "filename": "neural_nets.pdf",
            "chunk_index": 2,
            "doc_type": "pdf",
            "distance": 0.88
        },
        {
            "chunk_text": "Deep learning uses multiple layers of neural networks.",
            "filename": "deep_learning.txt",
            "chunk_index": 1,
            "doc_type": "txt",
            "distance": 0.82
        }
    ]


@pytest.fixture
def retriever_instance(mock_embedder, mock_weaviate):
    """Create a Retriever instance with mocked dependencies."""
    with patch('chatbot.pipeline.retriever.get_embedder_client', return_value=mock_embedder), \
         patch('chatbot.pipeline.retriever.get_weaviate_client', return_value=mock_weaviate):
        retriever = Retriever()
        return retriever


# ============ Test 1: Retriever Initialization ============

class TestRetrieverInitialization:
    """Test suite for Retriever initialization."""
    
    def test_retriever_initializes_with_clients(self):
        """Test that Retriever properly initializes embedder and vector DB clients."""
        mock_embedder = Mock()
        mock_weaviate = Mock()
        
        with patch('chatbot.pipeline.retriever.get_embedder_client', return_value=mock_embedder), \
             patch('chatbot.pipeline.retriever.get_weaviate_client', return_value=mock_weaviate):
            retriever = Retriever()

        # print(f"Actual: {retriever.embedder}")
        # print(f"Mock: {mock_embedder}")
        
        assert retriever.embedder is mock_embedder
        assert retriever.vecdb is mock_weaviate
    
    def test_get_retriever_returns_singleton(self):
        """Test that get_retriever() implements singleton pattern correctly."""
        with patch('chatbot.pipeline.retriever.get_embedder_client'), \
             patch('chatbot.pipeline.retriever.get_weaviate_client'):
            
            # Reset singleton
            from chatbot.pipeline import retriever
            retriever._retriever = None
            
            retriever1 = get_retriever()
            retriever2 = get_retriever()
        
        # Assert
        assert retriever1 is retriever2


# ============ Test 2: Retrieve Method - Happy Path ============

class TestRetrieveHappyPath:
    """Test suite for successful retrieval scenarios."""
    
    def test_retrieve_returns_formatted_sources(
        self, 
        retriever_instance, 
        sample_vector_db_results
    ):
        """Test that retrieve() returns properly formatted sources with all fields."""
        # Arrange
        query = "What is machine learning?"
        retriever_instance.vecdb.search.return_value = sample_vector_db_results
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        assert len(sources) == 3
        assert all(isinstance(source, dict) for source in sources)
        
        # Verify structure of each source
        for idx, source in enumerate(sources, start=1):
            assert source["source_id"] == idx
            assert "chunk_text" in source
            assert "filename" in source
            assert "chunk_index" in source
            assert "doc_type" in source
            assert "relevance_score" in source
            assert source["cited"] is False
    
    def test_retrieve_assigns_sequential_source_ids(
        self, 
        retriever_instance, 
        sample_vector_db_results
    ):
        """Test that source_id is assigned sequentially starting from 1."""
        # Arrange
        query = "Test query"
        retriever_instance.vecdb.search.return_value = sample_vector_db_results
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        source_ids = [s["source_id"] for s in sources]
        assert source_ids == [1, 2, 3]
    
    def test_retrieve_preserves_metadata_from_vector_db(
        self, 
        retriever_instance, 
        sample_vector_db_results
    ):
        """Test that all metadata from vector DB is preserved in output."""
        # Arrange
        query = "Test query"
        retriever_instance.vecdb.search.return_value = sample_vector_db_results
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        assert sources[0]["filename"] == "ml_intro.pdf"
        assert sources[0]["chunk_index"] == 0
        assert sources[0]["doc_type"] == "pdf"
        assert sources[0]["relevance_score"] == 0.95
        
        assert sources[1]["filename"] == "neural_nets.pdf"
        assert sources[1]["chunk_index"] == 2
        assert sources[1]["relevance_score"] == 0.88
    
    def test_retrieve_calls_embedder_with_query(self, retriever_instance):
        """Test that embedder is called with the correct query."""
        # Arrange
        query = "What is deep learning?"
        retriever_instance.vecdb.search.return_value = []
        
        # Act
        retriever_instance.retrieve(query)
        
        # Assert
        retriever_instance.embedder.embed_query.assert_called_once_with(query)
    
    def test_retrieve_calls_vecdb_with_correct_parameters(
        self, 
        retriever_instance,
        mock_settings
    ):
        """Test that vector DB search is called with correct parameters."""
        # Arrange
        query = "Test query"
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        retriever_instance.embedder.embed_query.return_value = query_embedding
        retriever_instance.vecdb.search.return_value = []
        
        # Act
        retriever_instance.retrieve(query)
        
        # Assert
        retriever_instance.vecdb.search.assert_called_once_with(
            query_vector=query_embedding,
            limit=mock_settings.max_sources,
            distance_threshold=mock_settings.similarity_threshold
        )


# ============ Test 3: Retrieve Method - Edge Cases ============

class TestRetrieveEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_retrieve_with_empty_results(self, retriever_instance):
        """Test that retrieve() handles empty vector DB results gracefully."""
        # Arrange
        query = "Unrelated query with no matches"
        retriever_instance.vecdb.search.return_value = []
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        assert sources == []
        assert isinstance(sources, list)
    
    def test_retrieve_with_single_result(self, retriever_instance):
        """Test retrieve with exactly one result."""
        # Arrange
        query = "Test query"
        single_result = [{
            "chunk_text": "Single result text",
            "filename": "single.pdf",
            "chunk_index": 0,
            "doc_type": "pdf",
            "distance": 0.92
        }]
        retriever_instance.vecdb.search.return_value = single_result
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        assert len(sources) == 1
        assert sources[0]["source_id"] == 1
        assert sources[0]["chunk_text"] == "Single result text"

    def test_retrieve_respects_max_sources_limit_and_similarity_threshold(
        self, 
        retriever_instance
    ):
        """Test that max_sources and similarity_threshold settings are passed to vector DB."""
        # Arrange
        query = "Test query"
        retriever_instance.vecdb.search.return_value = []

        with patch('chatbot.pipeline.retriever.settings') as mock_settings:
            mock_settings.max_sources = 5
            mock_settings.similarity_threshold = 0.85

            # Act
            retriever_instance.retrieve(query)
        
            # Assert
            call_args = retriever_instance.vecdb.search.call_args
            assert call_args.kwargs["limit"] == 5
            assert call_args.kwargs["distance_threshold"] == 0.85
    
    def test_retrieve_with_special_characters_in_query(self, retriever_instance):
        """Test that queries with special characters are handled correctly."""
        # Arrange
        query = "What is ML? (Machine Learning) & AI!"
        retriever_instance.vecdb.search.return_value = []
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        retriever_instance.embedder.embed_query.assert_called_once_with(query)
        assert sources == []
    
    def test_retrieve_with_very_long_query(self, retriever_instance):
        """Test that very long queries are processed without issues."""
        # Arrange
        query = "What is machine learning? " * 100  # Very long query
        retriever_instance.vecdb.search.return_value = []
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        retriever_instance.embedder.embed_query.assert_called_once()
        assert sources == []
    
    def test_retrieve_with_unicode_in_results(self, retriever_instance):
        """Test that unicode characters in results are preserved."""
        # Arrange
        query = "Test query"
        unicode_results = [{
            "chunk_text": "机器学习 is 機械学習 in different languages",
            "filename": "unicode_文件.pdf",
            "chunk_index": 0,
            "doc_type": "pdf",
            "distance": 0.90
        }]
        retriever_instance.vecdb.search.return_value = unicode_results
        
        # Act
        sources = retriever_instance.retrieve(query)
        
        # Assert
        assert "机器学习" in sources[0]["chunk_text"]
        assert "unicode_文件.pdf" in sources[0]["filename"]


# ============ Test 4: Error Handling ============

class TestRetrieveErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_retrieve_raises_exception_when_embedding_fails(self, retriever_instance):
        """Test that embedding failures raise exceptions properly."""
        # Arrange
        query = "Test query"
        retriever_instance.embedder.embed_query.side_effect = Exception("Embedding service unavailable")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            retriever_instance.retrieve(query)
        
        assert "Embedding service unavailable" in str(exc_info.value)
    
    def test_retrieve_raises_exception_when_vecdb_search_fails(self, retriever_instance):
        """Test that vector DB failures raise exceptions properly."""
        # Arrange
        query = "Test query"
        retriever_instance.vecdb.search.side_effect = Exception("Weaviate connection timeout")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            retriever_instance.retrieve(query)
        
        assert "Weaviate connection timeout" in str(exc_info.value)
    
    def test_retrieve_logs_error_on_embedding_failure(self, retriever_instance):
        """Test that embedding failures are logged."""
        # Arrange
        query = "Test query"
        retriever_instance.embedder.embed_query.side_effect = ValueError("Invalid input")
        
        # Act & Assert
        with patch('chatbot.pipeline.retriever.logger') as mock_logger:
            with pytest.raises(ValueError):
                retriever_instance.retrieve(query)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Embedding failed" in str(mock_logger.error.call_args)
    
    def test_retrieve_logs_error_on_retrieval_failure(self, retriever_instance):
        """Test that retrieval failures are logged."""
        # Arrange
        query = "Test query"
        retriever_instance.vecdb.search.side_effect = ConnectionError("DB unavailable")
        
        # Act & Assert
        with patch('chatbot.pipeline.retriever.logger') as mock_logger:
            with pytest.raises(ConnectionError):
                retriever_instance.retrieve(query)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Retrieval failed" in str(mock_logger.error.call_args)

