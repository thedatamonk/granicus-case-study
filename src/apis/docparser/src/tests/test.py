import asyncio
import httpx
import time
from pathlib import Path
import shutil
from typing import List

async def upload(client, url, file_paths, req_id):
    start = time.time()
    print(f"Request {req_id}: START @ {time.ctime(start)}")
    # Open all files for multipart upload
    file_handles = []
    files_data = []
    
    try:
        for file_path in file_paths:
            f = open(file_path, 'rb')
            file_handles.append(f)
            # Note: 'files' is the field name, needs to match FastAPI's List[UploadFile]
            files_data.append(('files', (file_path.name, f, 'application/octet-stream')))
        
        # Send all files in one request
        response = await client.post(url, files=files_data)
    
    finally:
        # Close all file handles
        for f in file_handles:
            f.close()
    
    duration = time.time() - start
    print(f"Request {req_id}: DONE in {duration:.2f}s - Response: {response.json()}")
    return duration

def create_file_copies(source_file: Path, num_copies: int, output_dir: Path) -> List[Path]:
    """Create multiple copies of a source file"""
    output_dir.mkdir(exist_ok=True)
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    print(f"Creating {num_copies} copies of '{source_file.name}'...")
    print(f"Source file size: {source_file.stat().st_size / (1024*1024):.2f} MB")
    
    file_paths = []
    for i in range(num_copies):
        # Create copy with numbered suffix
        copy_path = output_dir / f"{source_file.stem}_copy_{i}{source_file.suffix}"
        
        if not copy_path.exists():
            shutil.copy2(source_file, copy_path)
        
        file_paths.append(copy_path)
    
    print(f"✅ Created {len(file_paths)} file copies in '{output_dir}'\n")
    return file_paths

async def main():
    url = "http://localhost:8002/v1/ingest"
    NUM_CONCURRENT_REQUESTS = 10
    NUM_FILES_PER_REQUEST = 10
    FILE_SIZE_MB = 50

    # Create NUM_FILES_PER_REQUEST dummy files of size 30MB each
    # root_dir = Path(__file__).parent / "dummy_files"
    # root_dir.mkdir(exist_ok=True)
    # file_paths = []
    # for i in range(NUM_FILES_PER_REQUEST):
    #     file_path = root_dir / f"dummy_file_{i}.txt"
    #     if not file_path.exists():
    #         # create dummy file
    #         file_path.write_bytes(b'0' * (FILE_SIZE_MB * 1024 * 1024)) # 30 MB file
    #     file_paths.append(file_path)

    source_file = Path("/Users/rohil/rohil-workspace/career/interviews/granicus/problem_docs/technical_specs.txt")
    root_dir = Path(__file__).parent / "test_files"
    root_dir.mkdir(exist_ok=True)

    try:
        file_paths = create_file_copies(
            source_file=source_file,
            num_copies=NUM_FILES_PER_REQUEST,
            output_dir=root_dir
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Please update source_file path in the script to point to your real file!")
        return
    
    total_size_mb = sum(f.stat().st_size for f in file_paths) / (1024 * 1024)
    print(f"📦 Total size per request: {total_size_mb:.2f} MB")
    print(f"📦 Total size for all requests: {total_size_mb * NUM_CONCURRENT_REQUESTS:.2f} MB\n")


    # Pass to upload function
    # Scenario 1: Async client -> Async Server
    start_time = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [upload(client, url, file_paths, i+1) for i in range(NUM_CONCURRENT_REQUESTS)]
        durations = await asyncio.gather(*tasks)
    
    total = time.time() - start_time
    print(f"\n✅ Total time: {total:.2f}s")
    print(f"📊 Sum of individual times: {sum(durations):.2f}s")
    print(f"🚀 Speedup: {sum(durations) / total:.2f}x")

    # Scenario 2: Sync clinet -> Async Server
    # start_time = time.time()
    # async with httpx.AsyncClient(timeout=30) as client:
    #     # NOw send requests sequentially
    #     for i in range(NUM_CONCURRENT_REQUESTS):
    #         req_start_time = time.time()
    #         print(f"Request {i+1}: SENT @ {time.ctime(req_start_time)}")
    #         duration = await upload(client=client, url=url, file_paths=file_paths, req_id=i+1)

    # total = time.time() - start_time
    # print(f"✅ Total time: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())