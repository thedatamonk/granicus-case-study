import pytest
from fastapi import UploadFile, HTTPException
from io import BytesIO
from src.apis.docparser.src.docparser.helpers import (
    validate_files,
    extract_text_from_pdf,
    create_chunks_from_extraction,
    MAX_FILES,
    MAX_TOTAL_SIZE,
)


# ============ Fixtures ============

@pytest.fixture
def mock_pdf_content():
    """Create a minimal valid PDF content for testing."""
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""
    return pdf_content


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile for testing."""
    def _create_file(filename: str, content: bytes, size: int|None = None) -> UploadFile:
        file = UploadFile(
            filename=filename,
            file=BytesIO(content),
            size=size,
        )
        return file
    return _create_file


@pytest.fixture
def valid_pdf_files(mock_upload_file):
    """Create a list of valid PDF files."""
    return [
        mock_upload_file("test1.pdf", b"fake pdf content", 1000),
        mock_upload_file("test2.pdf", b"another pdf", 1000)
    ]


# ============ Test 1: File Validation ============

class TestValidateFiles:
    """Test suite for file validation logic."""
    
    def test_validate_files_success_with_valid_files(self, mock_upload_file):
        """Test that valid files pass validation."""
        # Arrange
        files = [
            mock_upload_file("doc1.pdf", b"x" * 1000),
            mock_upload_file("data.csv", b"y" * 2000),
            mock_upload_file("notes.txt", b"z" * 1500)
        ]
        
        # Act & Assert - should not raise any exception
        validate_files(files)
    
    def test_validate_files_raises_error_when_too_many_files(self, mock_upload_file):
        """Test that exceeding MAX_FILES raises HTTPException."""
        # Arrange
        files = [mock_upload_file(f"file{i}.pdf", b"content") for i in range(MAX_FILES + 1)]
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            validate_files(files)
        
        assert exc_info.value.status_code == 400
        assert f"Maximum {MAX_FILES} files allowed" in str(exc_info.value.detail)
    
    def test_validate_files_raises_error_for_invalid_extension(self, mock_upload_file):
        """Test that invalid file extensions raise HTTPException."""
        # Arrange
        files = [
            mock_upload_file("valid.pdf", b"content"),
            mock_upload_file("invalid.docx", b"content")  # Invalid extension
        ]
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            validate_files(files)
        
        assert exc_info.value.status_code == 400
        assert "invalid.docx" in str(exc_info.value.detail)
        assert "invalid type" in str(exc_info.value.detail).lower()
    
    def test_validate_files_raises_error_when_total_size_exceeds_limit(self, mock_upload_file):
        """Test that exceeding MAX_TOTAL_SIZE raises HTTPException."""
        # Arrange - Create files that exceed the total size limit
        large_content = b"x" * (MAX_TOTAL_SIZE // 2 + 1)
        files = [
            mock_upload_file("large1.pdf", large_content),
            mock_upload_file("large2.pdf", large_content)
        ]
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            validate_files(files)
        
        assert exc_info.value.status_code == 400
        assert "exceeds" in str(exc_info.value.detail).lower()


# ============ Test 2: PDF Text Extraction ============

class TestExtractTextFromPDF:
    """Test suite for PDF text extraction."""
    
    def test_extract_text_from_pdf_success(self, mock_pdf_content):
        """Test successful PDF text extraction returns correct structure."""
        # Act
        result = extract_text_from_pdf(mock_pdf_content)
        
        # Assert
        assert result["success"] is True
        assert "pages" in result
        assert "total_pages" in result
        assert isinstance(result["pages"], list)
        assert result["total_pages"] == len(result["pages"])
        
        # Verify page structure
        if result["pages"]:
            page = result["pages"][0]
            assert "page" in page
            assert "text" in page
            assert isinstance(page["page"], int)
            assert isinstance(page["text"], str)
    
    def test_extract_text_from_pdf_handles_invalid_content(self):
        """Test that invalid PDF content returns error structure."""
        # Arrange
        invalid_content = b"This is not a PDF"
        # Act
        result = extract_text_from_pdf(invalid_content)
        
        # Assert
        assert result["success"] is False
        assert "error" in result
        assert "extraction failed" in result["error"].lower()
    
    def test_extract_text_from_pdf_handles_empty_content(self):
        """Test that empty content returns error structure."""
        # Arrange
        empty_content = b""
        
        # Act
        result = extract_text_from_pdf(empty_content)
        
        # Assert
        assert result["success"] is False
        assert "error" in result


# ============ Test 3: Chunk Creation ============

class TestCreateChunksFromExtraction:
    """Test suite for chunk creation from extracted text."""
    
    def test_create_chunks_from_pdf_extraction(self):
        """Test chunk creation from PDF extraction results."""
        # Arrange
        extraction_result = {
            "success": True,
            "pages": [
                {"page": 1, "text": "This is page one content. " * 100},  # Long text
                {"page": 2, "text": "This is page two content. " * 100}
            ],
            "total_pages": 2
        }
        filename = "test.pdf"
        
        # Act
        chunks = create_chunks_from_extraction(extraction_result, filename)
        
        # Assert
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        
        # Verify chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == filename
            assert chunk["metadata"]["type"] == "pdf"
            assert "page" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0
    
    def test_create_chunks_from_csv_extraction(self):
        """Test chunk creation from CSV extraction results."""
        # Arrange
        extraction_result = {
            "success": True,
            "headers": ["Name", "Age", "City"],
            "rows": [
                "Name: John | Age: 30 | City: NYC",
                "Name: Jane | Age: 25 | City: LA",
                "Name: Bob | Age: 35 | City: SF",
                "Name: Alice | Age: 28 | City: Seattle"
            ],
            "total_rows": 4
        }
        filename = "data.csv"
        
        # Act
        chunks = create_chunks_from_extraction(extraction_result, filename)
        
        # Assert
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == filename
            assert chunk["metadata"]["type"] == "csv"
            assert "row_start" in chunk["metadata"]
            assert "row_end" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            # Verify headers are included
            assert "Headers:" in chunk["text"]
    
    def test_create_chunks_returns_empty_for_failed_extraction(self):
        """Test that failed extraction results return empty chunks."""
        # Arrange
        extraction_result = {
            "success": False,
            "error": "Extraction failed"
        }
        filename = "failed.pdf"
        
        # Act
        chunks = create_chunks_from_extraction(extraction_result, filename)
        
        # Assert
        assert chunks == []
        assert isinstance(chunks, list)
    
    def test_create_chunks_from_txt_extraction(self):
        """Test chunk creation from TXT extraction results."""
        # Arrange
        extraction_result = {
            "success": True,
            "paragraphs": [
                "This is the first paragraph with some content.",
                "This is the second paragraph with more information.",
                "And here is a third paragraph to test chunking."
            ],
            "total_paragraphs": 3
        }
        filename = "notes.txt"
        
        # Act
        chunks = create_chunks_from_extraction(extraction_result, filename)
        
        # Assert
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == filename
            assert chunk["metadata"]["type"] == "txt"
            assert "chunk_index" in chunk["metadata"]