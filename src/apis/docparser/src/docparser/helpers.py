import hashlib
import re
from pathlib import Path
from typing import List

from docparser.clients.llm_client import get_llm_client
from fastapi import HTTPException, UploadFile
from langchain_text_splitters import MarkdownHeaderTextSplitter

MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILES = 20
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md"}


def get_doc_id(filename: str, filecontent: bytes, doctype: str) -> str:
    filename = filename.split(".")[0]
    content_hash = hashlib.md5(filecontent).hexdigest()[:8]
    return f"{doctype}-{filename}-{content_hash}"


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

        # # Suppress the SWIG deprecation warning
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=DeprecationWarning)
        #     import pymupdf

        # doc = pymupdf.open(stream=content, filetype="pdf")
        
        # pages = []
        # for page_num, page in enumerate(doc, 1):
        #     text = page.get_text()
        #     pages.append({
        #         "page": page_num,
        #         "text": text.strip()
        #     })
        
        # doc.close()
        # return {
        #     "success": True,
        #     "content": pages,
        # }
        from io import BytesIO

        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(stream=BytesIO(content), filetype="pdf")
        md_content = pymupdf4llm.to_markdown(doc)

        doc.close()

        return {
            "success": True,
            "content": md_content
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"PDF extraction failed: {str(e)}"
        }


def extract_text_from_csv(content: bytes) -> dict:
    """Extract text from CSV with structure preservation & convert them to a JSON string."""
    try:
        import json
        from io import BytesIO

        import chardet
        import pandas as pd

        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Read CSV
        df = pd.read_csv(BytesIO(content), encoding=encoding)

        # Convert dataframe to list of dictionary (one dict per row)
        # which we convert later to JSON string
        records = df.to_dict(orient="records")
        json_str = json.dumps(records, ensure_ascii=False)
        
        return {
            "success": True,
            "content": json_str
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"CSV extraction failed: {str(e)}",
        }


def extract_text_from_txt(content: bytes) -> dict:
    """Extract raw text from TXT with encoding detection."""
    try:
        import chardet

        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Try detected encoding, fallback to utf-8
        text = content.decode(encoding)

        return {
            "success": True,
            "content": text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"TXT extraction failed: {str(e)}",
        }


def extract_text(content: bytes, filename: str) -> dict:
    """Route to appropriate extractor based on file type."""
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(content)
    elif ext == ".csv":
        return extract_text_from_csv(content)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(content)
    else:
        return {"success": False, "error": "Unsupported file type"}


def create_chunks(extraction_result: dict, filename: str) -> List[dict] | dict:
    # Chunking router to decide chunking strategy based on filetype
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        return chunk_text_from_pdf(extraction_result["content"])
    elif ext == ".csv":
        return chunk_text_from_csv(extraction_result["content"])
    elif ext == ".txt":
        return chunk_text_from_txt(extraction_result["content"])
    elif ext == ".md":
        return chunk_text_from_md(extraction_result["content"])
    else:
        return {"success": False, "error": "Unsupported file type"}


def chunk_text_from_pdf(content: str) -> List[dict] | dict:
    try:
        headers_to_split_on = [
            ("##", "Section Name"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        splits = markdown_splitter.split_text(content)

        # Convert each split into the chunk format
        chunks = []
        for split in splits:
            # if metadata is not None or empty -> We don't treat it as a chunk
            if split.metadata:
                chunks.append(
                    {
                        "content": split.page_content,
                        "metadata": split.metadata
                    }
                )

        return chunks
    except Exception as e:
        return {
            "success": False, "error": f"The input text from the given .PDF file is not in the format that this chunking strategy is implemented for.\nGot this error: {e}"
        }
    

def chunk_text_from_txt(text: str) -> List[dict] | dict:
    """
    Chunk text from Granicus .txt files
    NOTE: These files are in a certain format and hence this strategy has been implemented
    to cater to such .txt files only
    """
    try:
        chunks = _split_document_by_headings(text)
        return chunks
    except Exception as e:
        return {
            "success": False, "error": f"The input text from the given .TXT file is not in the format that this chunking strategy is implemented for.\nGot this error: {e}"
        }


def chunk_text_from_md(text: str) -> List[dict] | dict:
    """
    Chunk text from Granicus .md files
    NOTE: These files are in a certain format and hence this strategy has been implemented
    to cater to such .md files only.
    """
    try:
        chunks = _split_product_markdown_doc(text)
        return chunks
    except Exception as e:
        return {
            "success": False, "error": f"The input text from the given .MD file is not in the format that this chunking strategy is implemented for.\nGot this error: {e}"
        }


def dict_to_readable_string(d: dict, indent: int = 0) -> str:
    """Convert nested dict to indented hierarchical string"""
    lines = []
    prefix = "  " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dict_to_readable_string(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    lines.append(f"{prefix}  [{i}]:")
                    lines.append(dict_to_readable_string(item, indent + 2))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return '\n'.join(lines)

def chunk_text_from_csv(text: str) -> List[dict] | dict:
    """
    Chunk text from Granicus .csv files
    The text parameter is a valid JSON string that represents that data contained in the .csv file
    In this function, we will use an LLM to 
    """
    try:
        llm_client = get_llm_client()
        json_obj = llm_client.generate(text)

        # for each entry in json_obj, create a chunk and its metadata
        chunks = []
        for obj_label, obj_data in json_obj.items():
            chunk_content = f"**{obj_label}**\n\n"
            chunk_content += dict_to_readable_string(obj_data)
            chunk = {
                "content": chunk_content,
                "metadata": {
                    "chunk_label": obj_label
                }
            }

            chunks.append(chunk)

        return chunks
    except Exception as e:
        return {
            "success": False, "error": f"The extracted JSON string from the given .CSV file is not in the format that this LLM based chunking strategy is implemented for.\nGot this error: {e}"
        }

    
def _split_product_markdown_doc(text: str) -> List[dict]:
    # Extract document name from first line
    lines = text.strip().split('\n')
    document_name = lines[0].replace('#', '').strip() if lines else "Unknown Document"
    
    # Split by --- separator
    chunks_raw = text.split('\n---\n')
    
    # Skip the first chunk (header with title and subtitle)
    chunks_raw = chunks_raw[1:] if len(chunks_raw) > 1 else chunks_raw
    
    chunks = []
    for chunk_content in chunks_raw:
        chunk_content = chunk_content.strip()
        if not chunk_content:
            continue
            
        # Extract metadata only if chunk content found
        metadata = _extract_chunk_metadata(chunk_content, document_name)
        chunks.append({
            'content': chunk_content,
            'metadata': metadata
        })

    return chunks
    
def _extract_chunk_metadata(content: str, document_name: str) -> dict | None:
    lines = content.split('\n')
    
    # Find first non-empty line with heading
    first_heading = None
    first_heading_idx = None
    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#'):
            first_heading = line
            first_heading_idx = idx
            break
    
    if not first_heading:
        return None
    
    # Skip container headings like "### PRODUCT PORTFOLIO"
    if 'PRODUCT PORTFOLIO' in first_heading:
        # Look for the next heading (should be the actual product)
        for line in lines[first_heading_idx + 1:]:
            line = line.strip()
            if line.startswith('##'):
                first_heading = line
                break
    
    # Determine chunk type and extract info
    chunk_type = None
    product_number = None
    product_name = None
    section_heading = None
    
    # Check if it's a product (pattern: ## 1. PRODUCT_NAME or ## 1. **PRODUCT_NAME**)
    product_pattern = r'^##\s+(\d+)\.\s+(?:\*\*)?(.*?)(?:\*\*)?$'
    product_match = re.match(product_pattern, first_heading)
    
    if product_match:
        chunk_type = 'product'
        product_number = int(product_match.group(1))
        product_name = product_match.group(2).strip()
        section_heading = product_name
    else:
        # It's a supporting section
        chunk_type = 'supporting_section'
        # Remove all # and clean up
        section_heading = re.sub(r'^#+\s*', '', first_heading).strip()
    
    # Extract subsections (all ### headings)
    subsections = []
    for line in lines:
        line = line.strip()
        if line.startswith('###'):
            subsection = re.sub(r'^###\s*', '', line).strip()
            subsections.append(subsection)
    
    # Check if chunk has pricing information
    has_pricing = 'Pricing Tiers' in content or 'pricing' in content.lower()
    
    # Build metadata dictionary
    metadata = {
        'document_name': document_name,
        'chunk_type': chunk_type,
        'section_heading': section_heading,
        'has_pricing': has_pricing,
        'subsections': subsections,
    }
    
    # Add product-specific fields
    if chunk_type == 'product':
        metadata['product_number'] = product_number
        metadata['product_name'] = product_name
    
    return metadata

def _split_document_by_headings(text: str) -> List[dict]:
    # Extract document name (first line)
    lines = text.strip().split('\n')
    document_name = lines[0].strip() if lines else "Unknown Document"
    
    # Pattern to match === HEADING ===
    heading_pattern = r'^===\s+(.+?)\s+===$'
    
    chunks = []
    current_heading = None
    current_content = []

    for line in lines[1:]:  # Skip first line (document name)
        # Check if line is a heading
        match = re.match(heading_pattern, line.strip())
        
        if match:
            # Save previous chunk if exists
            if current_heading is not None:
                chunks.append({
                    'content': '\n'.join(current_content).strip(),
                    'metadata': {
                        'document_name': document_name,
                        'section_heading': current_heading,
                    }
                })
            
            # Start new chunk
            current_heading = match.group(1).strip()
            current_content = []
        else:
            # Add line to current content
            if current_heading is not None:  # Only add if we've started a section
                current_content.append(line)
    
    # Add the last chunk
    if current_heading is not None:
        chunks.append({
            'content': '\n'.join(current_content).strip(),
            'metadata': {
                'document_name': document_name,
                'section_heading': current_heading,
            }
        })

    return chunks

    return chunks
