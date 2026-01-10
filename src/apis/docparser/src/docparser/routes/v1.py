from fastapi import APIRouter, UploadFile, BackgroundTasks, HTTPException
from loguru import logger
from typing import List
from docparser.helpers import validate_files, extract_text, create_chunks, get_doc_id
import json
from datetime import datetime, timezone
import uuid
from docparser.settings import get_settings
from docparser.clients.embedding_client import get_embedder_client
from docparser.clients.weaviate_client import get_weaviate_client
from pathlib import Path

settings = get_settings()

router = APIRouter(prefix="/v1")

# in-memoy storage for file ingestion jobs
jobs = {}

# ============ Background Processing ============
async def process_files_task(job_id: str, files_data: List[dict]):
    """Background task to process files."""
    jobs[job_id]["status"] = "processing"
    results = []

    # Get embedder and weaviate client
    embedder_client = get_embedder_client()
    weaviate_client = get_weaviate_client()
    
    for file_data in files_data:
        filename = file_data["filename"]
        content = file_data["content"]
        doctype = Path(filename).suffix.lower().strip(".")
        docid = get_doc_id(file_data['filename'], file_data['content'], doctype)
        
        try:
            # Extract text
            logger.info(f"Extracting text from {filename}...")
            extraction = extract_text(content, filename)
            
            if not extraction["success"]:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": extraction["error"],
                })
                continue
            
            # Create chunks
            logger.info(f"Chunking text from {filename}...")
            chunks = create_chunks(extraction, filename)
            
            # Store chunks locally
            output_path = Path(settings.processed_docs_dir) / job_id / f"{Path(filename).stem}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storing raw data @ {output_path}...")
            
            with open(output_path, 'w') as f:
                json.dump({
                    "filename": filename,
                    "extraction": extraction,
                    "chunks": chunks,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
            
            # Call the embedding service to generate the embeddings for each chunks
            logger.info(f"Generating embeddings for {filename}...")
            chunks_texts = [chunk["content"] for chunk in chunks]
            try:
                embeddings = embedder_client.generate_embeddings(texts=chunks_texts)
                logger.info(f"Generated embeddings for {len(chunks_texts)} chunks.")
            except Exception as e:
                logger.error(f"Embedding generation failed for {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": f"Embedding generation failed: {str(e)}",
                })
                continue

            # Combine embeddings with doc meta data
            docs_and_embeddings = []
            for _, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                docs_and_embeddings.append({
                    "properties": {
                        "chunk_text": chunk["content"],
                        "metadata": json.dumps(chunk["metadata"]),
                        "source": docid,
                        "doc_type": doctype,
                        "job_id": job_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "vector": embedding
                })


            # Insert embeddings into weaviate vector db
            logger.info(f"Inserting {len(embeddings)} embeddings into Weaviate vectordb for {filename}...")
            try:
                weaviate_client.insert_chunks(docs_and_embeddings)
                logger.info("Successfully stored chunks.")
            except Exception as e:
                logger.error(f"Weaviate insertion failed for {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": f"Weaviate insertion failed: {str(e)}",
                })
                continue

            results.append({
                "filename": filename,
                "status": "success",
                "error": None
            })
            
        except Exception as e:
            results.append({
                "filename": filename,
                "status": "failed",
                "error": str(e),
            })
    
    # Update job status
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["results"] = results
    jobs[job_id]["processed_files"] = len(results)
    jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/ingest")
async def ingest(files: List[UploadFile], background_tasks: BackgroundTasks):

    logger.info("Validating documents...")
    validate_files(files)

    # generate file ingestion job id
    job_id = str(uuid.uuid4())

    # Read file contents
    logger.info("Reading documents...")
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append({
            "filename": file.filename,
            "content": content
        })

    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(files),
        "processed_files": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": []
    }

    # Queue background task
    logger.info(f"Creating JOB: {job_id}")
    background_tasks.add_task(process_files_task, job_id, files_data)
    logger.info("Job created.")

    return {
        "job_id": job_id,
        "status": "queued",
        "files_submitted": len(files),
        "message": "Processing started"
    }

@router.get("/ingest/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an ingestion job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    return jobs[job_id]