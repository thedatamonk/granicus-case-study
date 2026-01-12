import pytest
from fastapi import UploadFile, HTTPException
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import hashlib

from docparser.helpers import (
    validate_files,
    get_doc_id,
    extract_text_from_csv,
    chunk_text_from_csv,
    _split_document_by_headings
)


# ============ Fixtures ============

@pytest.fixture
def valid_upload_files():
    """Create valid mock UploadFile objects."""
    mock_files = []
    for i in range(3):
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = f"test_{i}.csv"
        mock_file.file = BytesIO(b"test content " * 100)
        mock_files.append(mock_file)
    return mock_files


@pytest.fixture
def sample_csv_content():
    """Sample CSV content as bytes."""
    return b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"


@pytest.fixture
def sample_heading_document():
    """Sample document with === HEADING === format."""
    return """Sample Document Title
=== Introduction ===
This is the introduction section.
It has multiple lines.

=== Methods ===
This describes the methods.
More details here.

=== Conclusion ===
Final thoughts.
"""


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for CSV chunking."""
    return {
        "Product_A": {
            "name": "Widget Pro",
            "price": 99.99,
            "features": ["fast", "reliable"]
        },
        "Product_B": {
            "name": "Gadget Plus",
            "price": 149.99,
            "features": ["premium", "durable"]
        }
    }


class TestDocumentUtils:
    # ===== Validate Files =====
    
    def test_validate_files_passes_all_constraints(self, valid_upload_files):
        """Test that valid files pass all validation checks."""
        # Act & Assert - should not raise HTTPException
        try:
            validate_files(valid_upload_files)
        except HTTPException:
            pytest.fail("validate_files raised HTTPException unexpectedly")
    
    # ===== Get Doc ID =====
    
    def test_get_doc_id_generates_consistent_hash(self):
        """Test that doc_id is deterministic and includes all components."""
        # Arrange
        filename = "test_document.pdf"
        content = b"sample file content"
        doctype = "invoice"
        
        # Act
        doc_id = get_doc_id(filename, content, doctype)
        
        # Assert
        assert doc_id.startswith(f"{doctype}-test_document-")
        expected_hash = hashlib.md5(content).hexdigest()[:8]
        assert doc_id.endswith(expected_hash)
        
        # Verify consistency
        doc_id_2 = get_doc_id(filename, content, doctype)
        assert doc_id == doc_id_2
    
    # ===== Extract Text from CSV =====
    
    def test_extract_text_from_csv_converts_to_json(self, sample_csv_content):
        """Test CSV extraction with encoding detection and JSON conversion."""
        # Act
        result = extract_text_from_csv(sample_csv_content)
        
        # Assert
        assert result["success"] is True
        assert "content" in result
        
        import json
        parsed = json.loads(result["content"])
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0] == {"name": "Alice", "age": 30, "city": "NYC"}
        assert parsed[1] == {"name": "Bob", "age": 25, "city": "LA"}
    
    # ===== Split Document by Headings =====
    
    def test_split_document_by_headings_creates_chunks(self, sample_heading_document):
        """Test document splitting with === HEADING === pattern."""
        # Act
        chunks = _split_document_by_headings(sample_heading_document)
        
        # Assert
        assert len(chunks) == 3
        
        # Verify first chunk
        assert chunks[0]["metadata"]["section_heading"] == "Introduction"
        assert "introduction section" in chunks[0]["content"].lower()
        
        # Verify second chunk
        assert chunks[1]["metadata"]["section_heading"] == "Methods"
        assert "methods" in chunks[1]["content"].lower()
        
        # Verify third chunk
        assert chunks[2]["metadata"]["section_heading"] == "Conclusion"
        assert "final thoughts" in chunks[2]["content"].lower()
        
        # Verify all have document name
        for chunk in chunks:
            assert chunk["metadata"]["document_name"] == "Sample Document Title"
    
    # ===== Chunk Text from CSV =====
    
    @patch('docparser.helpers.get_llm_client')
    def test_chunk_text_from_csv_creates_chunks_from_llm(
        self, 
        mock_get_llm_client,
        mock_llm_response
    ):
        """Test CSV chunking with mocked LLM response."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.generate.return_value = mock_llm_response
        mock_get_llm_client.return_value = mock_llm
        
        json_string = '[{"id": 1, "data": "test"}]'
        
        # Act
        result = chunk_text_from_csv(json_string)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Verify first chunk
        assert "**Product_A**" in result[0]["content"]
        assert "Widget Pro" in result[0]["content"]
        assert result[0]["metadata"]["chunk_label"] == "Product_A"
        
        # Verify second chunk
        assert "**Product_B**" in result[1]["content"]
        assert "Gadget Plus" in result[1]["content"]
        assert result[1]["metadata"]["chunk_label"] == "Product_B"
        
        # Verify LLM called correctly
        mock_llm.generate.assert_called_once_with(json_string)