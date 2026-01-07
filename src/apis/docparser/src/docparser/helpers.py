from typing import List
from fastapi import UploadFile
from pathlib import Path
from fastapi import HTTPException
import warnings

MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILES = 20
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt"}

def validate_files(files: List[UploadFile]) -> None:
    """Validate uploaded files."""
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Maximum {MAX_FILES} files allowed")
    
    total_size = 0
    for file in files:
        # Check extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400, 
                f"File '{file.filename}' has invalid type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check size (read file to get actual size)
        content = file.file.read()
        file.file.seek(0)  # Reset for later use
        total_size += len(content)
    
    if total_size > MAX_TOTAL_SIZE:
        raise HTTPException(400, f"Total file size exceeds {MAX_TOTAL_SIZE // (1024*1024)}MB limit")


# ============ Text Extraction ============
def extract_text_from_pdf(content: bytes) -> dict:
    """Extract text from PDF with metadata."""
    try:

        # Suppress the SWIG deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pymupdf
        # import pymupdf
        doc = pymupdf.open(stream=content, filetype="pdf")
        
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            pages.append({
                "page": page_num,
                "text": text.strip()
            })
        
        doc.close()
        return {
            "success": True,
            "pages": pages,
            "total_pages": len(pages)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"PDF extraction failed: {str(e)}"
        }


def extract_text_from_csv(content: bytes, filename: str) -> dict:
    """Extract text from CSV with structure preservation."""
    try:
        import pandas as pd
        import chardet
        from io import BytesIO
        
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Read CSV
        df = pd.read_csv(BytesIO(content), encoding=encoding)
        
        # Convert to structured text
        rows = []
        headers = df.columns.tolist()
        
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in headers])
            rows.append(row_text)
        
        return {
            "success": True,
            "headers": headers,
            "rows": rows,
            "total_rows": len(rows)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"CSV extraction failed: {str(e)}"
        }


def extract_text_from_txt(content: bytes, filename: str) -> dict:
    """Extract text from TXT with encoding detection."""
    try:
        import chardet
        
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Try detected encoding, fallback to utf-8
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            text = content.decode('utf-8', errors='ignore')
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            "success": True,
            "paragraphs": paragraphs,
            "total_paragraphs": len(paragraphs)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"TXT extraction failed: {str(e)}"
        }


def extract_text(content: bytes, filename: str) -> dict:
    """Route to appropriate extractor based on file type."""
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(content, filename)
    elif ext == ".csv":
        return extract_text_from_csv(content, filename)
    elif ext == ".txt":
        return extract_text_from_txt(content, filename)
    else:
        return {"success": False, "error": "Unsupported file type"}


def chunk_text_fn(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    TODO: We need to work on this
    Simple token-based chunking with overlap.
    For production, use proper tokenizer (tiktoken, transformers).
    """
    # Simple word-based approximation (1 token ≈ 0.75 words)
    words = text.split()
    word_chunk_size = int(chunk_size * 0.75)
    word_overlap = int(overlap * 0.75)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + word_chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - word_overlap
        
        if start >= len(words):
            break
    
    return chunks

def create_chunks_from_extraction(extraction_result: dict, filename: str) -> List[dict]:
    """Create chunks from extracted text with metadata."""
    chunks = []
    
    if not extraction_result.get("success"):
        return []
    
    # PDF: chunk each page
    if "pages" in extraction_result:
        for page in extraction_result["pages"]:
            page_chunks = chunk_text_fn(page["text"])
            for idx, chunk in enumerate(page_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page["page"],
                        "chunk_index": idx,
                        "type": "pdf"
                    }
                })
    
    # CSV: chunk rows with headers
    elif "rows" in extraction_result:
        headers_text = " | ".join(extraction_result["headers"])
        rows = extraction_result["rows"]
        
        # Group rows into chunks
        rows_per_chunk = 3  # Configurable
        for i in range(0, len(rows), rows_per_chunk):
            chunk_rows = rows[i:i + rows_per_chunk]
            chunk_text = f"Headers: {headers_text}\n" + "\n".join(chunk_rows)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "row_start": i,
                    "row_end": min(i + rows_per_chunk, len(rows)),
                    "chunk_index": i // rows_per_chunk,
                    "type": "csv"
                }
            })
    
    # TXT: chunk paragraphs
    elif "paragraphs" in extraction_result:
        full_text = "\n\n".join(extraction_result["paragraphs"])
        text_chunks = chunk_text_fn(full_text)
        
        for idx, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": idx,
                    "type": "txt"
                }
            })
    
    return chunks