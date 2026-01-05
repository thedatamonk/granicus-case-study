from fastapi import APIRouter, UploadFile, BackgroundTasks, HTTPException
from loguru import logger
from typing import List
from docparser.helpers import validate_files, extract_text, create_chunks_from_extraction
import json
from datetime import datetime, timezone
import uuid
from docparser.settings import get_settings
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
    
    for file_data in files_data:
        filename = file_data["filename"]
        content = file_data["content"]
        
        try:
            # Extract text
            extraction = extract_text(content, filename)
            
            if not extraction["success"]:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": extraction["error"],
                    "chunks_created": 0
                })
                continue
            
            # Create chunks
            chunks = create_chunks_from_extraction(extraction, filename)
            
            # Store chunks locally
            output_path = Path(settings.processed_docs_dir) / job_id / f"{Path(filename).stem}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    "filename": filename,
                    "extraction": extraction,
                    "chunks": chunks,
                    "timestamp": datetime.utcnow().isoformat()
                }, f, indent=2)
            
            # TODO: Generate embeddings and store in RedisVL
            # embeddings = generate_embeddings([c["text"] for c in chunks])
            # store_in_redis(chunks, embeddings)
            
            results.append({
                "filename": filename,
                "status": "success",
                "chunks_created": len(chunks),
                "error": None
            })
            
        except Exception as e:
            results.append({
                "filename": filename,
                "status": "failed",
                "error": str(e),
                "chunks_created": 0
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