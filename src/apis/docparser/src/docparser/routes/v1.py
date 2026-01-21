import asyncio
import json
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from docparser.clients.embedding_client import get_embedder_client
from docparser.clients.weaviate_client import get_weaviate_client
from docparser.helpers import (create_chunks, extract_text, get_doc_id,
                               validate_files)
from docparser.settings import get_settings
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from loguru import logger

settings = get_settings()

router = APIRouter(prefix="/v1")

# in-memoy storage for file ingestion jobs
jobs = {}

# CPU Pool and I/O Pool
cpu_pool = ProcessPoolExecutor(max_workers=4)   # For CPU-bound work
io_pool = ThreadPoolExecutor(max_workers=20)    # For I/O-bound work


async def process_single_file(file_data: Dict, job_id: str, embedder_client, weaviate_client):
    """Process a single file asynchronously."""

    filename = file_data["filename"]
    content = file_data["content"]
    doctype = Path(filename).suffix.lower().strip(".")
    
    try:
        loop = asyncio.get_event_loop()
        
        # CPU-bound operations in process pool
        docid = await loop.run_in_executor(
            cpu_pool,
            get_doc_id, file_data['filename'], file_data['content'], doctype
        )

        logger.info(f"Extracting text from {filename}...")
        extraction = await loop.run_in_executor(
            cpu_pool,
            extract_text, content, filename
        )

        if not extraction["success"]:
            return {
                "filename": filename,
                "status": "failed",
                "error": extraction["error"],
            }
        
        logger.info(f"Chunking text from {filename}...")

        chunks = await loop.run_in_executor(
            cpu_pool,
            create_chunks, extraction, filename
        )

        # Generate embeddings (I/O-bound - API call)
        logger.info(f"Generating embeddings for {filename}...")
        chunks_texts = [chunk["content"] for chunk in chunks]

        try:
            embeddings = await loop.run_in_executor(
                io_pool,
                embedder_client.generate_embeddings, chunks_texts
            )
            logger.info(f"Generated embeddings for {len(chunks_texts)} chunks.")
        except Exception as e:
            logger.error(f"Embedding generation failed for {filename}: {e}")
            return {
                "filename": filename,
                "status": "failed",
                "error": f"Embedding generation failed: {str(e)}",
            }
        
        # Prepare data for Weaviate
        docs_and_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
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

        # Insert to Weaviate (I/O-bound)
        logger.info(f"Inserting {len(embeddings)} embeddings into Weaviate for {filename}...")
        try:
            await loop.run_in_executor(
                io_pool,
                weaviate_client.insert_chunks, docs_and_embeddings
            )
            logger.info(f"Successfully stored chunks for {filename}.")
        except Exception as e:
            logger.error(f"Weaviate insertion failed for {filename}: {e}")
            return {
                "filename": filename,
                "status": "failed",
                "error": f"Weaviate insertion failed: {str(e)}",
            }
        return {
            "filename": filename,
            "status": "success",
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return {
            "filename": filename,
            "status": "failed",
            "error": str(e),
        }
    

# ============ Background Processing ============
async def process_files_task(job_id: str, files: List[UploadFile]):
    """Background task to process files."""
    jobs[job_id]["status"] = "processing"
    results = []

    # Read the file data here
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append({
            "filename": file.filename,
            "content": content
        })
    
    logger.info("All files read.")
    # Get embedder and weaviate client
    embedder_client = get_embedder_client()
    weaviate_client = get_weaviate_client()
    
    # Process all files in parallel
    tasks = [
        process_single_file(file_data, job_id, embedder_client, weaviate_client)
        for file_data in files_data
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions from gather
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "filename": files_data[i]["filename"],
                "status": "failed",
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    # Update job status
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["results"] = processed_results
    jobs[job_id]["processed_files"] = len(processed_results)
    jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Job {job_id} completed. Processed {len(processed_results)} files.")


@router.post("/ingest")
async def ingest(files: List[UploadFile], background_tasks: BackgroundTasks):

    logger.info("Validating documents...")
    validate_files(files)

    # generate file ingestion job id
    job_id = str(uuid.uuid4())

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
    # NOTE: Commented now for testing
    logger.info(f"Creating JOB: {job_id}")
    background_tasks.add_task(process_files_task, job_id, files)
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
    # NOTE: This endpoint is blocked by the execution of /ingest endpoint
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    return jobs[job_id]    return jobs[job_id]