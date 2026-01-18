from locust import HttpUser, task, between
from pathlib import Path
import random


class DocIngestUser(HttpUser):
    # Wait 1-3 seconds b/w requests per user
    wait_time = between(1,1)

    def on_start(self) -> None:
        """Called once when user starts - prepare test files"""
        self.test_file = Path("/Users/rohil/rohil-workspace/career/interviews/granicus/problem_docs/technical_specs.txt")

        if not self.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file}")
        
    @task(3)        # Weight 3 - will run 3x more often than status check
    def upload_files(self):
        """Test the /ingest endpoint"""
        # Simulate user uploading 10 files per request
        files = []
        for i in range(10):
            with open(self.test_file, 'rb') as f:
                content = f.read()
                files.append(
                    ('files', (f'test_file_{i}.txt', content, 'text/plain'))
                )
        # Send the request
        with self.client.post(
            "/v1/ingest",
            files=files,
            catch_response=True,
            name="/v1/ingest [10 files]"  # Custom name in UI
        ) as response:
            if response.status_code == 200:
                # Store job_id for status checks
                self.job_id = response.json().get("job_id")
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)  # Weight 1 - runs less frequently
    def check_status(self):
        """Test the /ingest/status/{job_id} endpoint"""
        if hasattr(self, 'job_id') and self.job_id:
            with self.client.get(
                f"/v1/ingest/status/{self.job_id}",
                catch_response=True,
                name="/v1/ingest/status/{job_id}"
            ) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 404:
                    # Job might not exist yet - that's okay
                    response.success()
                else:
                    response.failure(f"Got status code {response.status_code}")