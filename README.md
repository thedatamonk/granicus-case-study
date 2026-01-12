# RAG chatbot system

A scalable production-ready RAG chatbot system that can answer questions about Granicus products and services. The system must handle diverse document formats, provide accurate information, and maintain strict adherence to the provided knowledge base.

## Key design decisions
1. The requirement of this problem statement was to build a RAG chatbot with **production-ready architecture and proper scaling**. Given this requirement, I immediately decided to structure my business logic into microservices. This will help us improve, test and deploy each component independently. 
2. I also wanted to make the building & running this project easier for any other person. Hence I used docker and docker compose.
3. The internal structure of each microservice may appear overkill for the current problem size. However, this was done deliberately to demonstrate how the same codebase can scale to a real production system. The separation between routing, business logic, and configuration makes it easier to add new features, swap components, and test individual pieces in isolation without refactoring the entire service.

## Note on submission timeline
The submission took slightly longer than initially expected. This was a deliberate choice. Given the breadth of the problem, I chose to prioritize a clean, extensible, and production-oriented design over a rushed implementation.

Instead of submitting a minimal or partially working solution, I spent additional time ensuring that the system has clear component boundaries, proper initialization flow, testability, and realistic scaling patterns. This extra time allowed me to deliver something closer to how I would actually build and ship such a system in a real production environment.

## Core components

- **Document Parser**: Ingests documents in a variety of formats and chunks them using custom chunking logic.
- **Embedder**: Create vector embeddings of the input texts using sentence transformers.
- **Vector Database**: Weaviate vector database for high-performance vector similarity search
- **Retriever**:  Retrieves documents from vector database based on distance from the query vector.
- **Reranker**: Reranks retrieved documents using Jina reranker model. 
- **Chatbot**: This is where all the components come together and make the RAG pipeline

## Architecture

The system consists of five main components:

### 1. Embedder Service (Port 8001)

- Generates dense vector embeddings from text using sentence transformer models
- Built with: `sentence-transformers`, `fastapi`
- Model: Configurable pre-trained transformer models

### 2. Docparser Service (Port 8002)

- Provides an endpoint to chunk and stores docs.
- The currently supported docs are (.pdf, .csv, .txt and .md)
- Built with: `openai`, `fastapi`

### 3. Weaviate Vector Database (Port 8080)

- Stores document embeddings and metadata
- Provides high-performance vector similarity search
- Built with `weaviate`

### 4. Reranker (Runs as part of chatbot service)

- Re-ranks search results using cross-encoder models for improved relevance
- Built with: `transformers`, `fastapi`
- Provides fine-grained relevance scoring

### 5. Chatbot Service

- Provides the main API endpoint for the chatbot
- Orchestrates embedding generation, vector search, and re-ranking
- Built with: `fastapi`, `weaviate`, `transformers` and `openai`
- Integrates with embedder and vectordb services

## Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, high-performance Python web framework
- **ML Models**:
  - [Sentence Transformers](https://www.sbert.net/) - Dense embedding generation
  - [Hugging Face Transformers](https://huggingface.co/transformers/) - Re-ranking models
- **Vector Database**: [Redis Stack](https://redis.io/docs/stack/) with vector similarity search
- **Package Management**: [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- **Containerization**: Docker & Docker Compose
- **Testing**: pytest

## Project Structure

```bash
.
├── src/apis
│   ├── chatbot/           # Chatbot service
│   ├── docparser/         # Docparsing service to ingest, parse and chunk docs
│   └── embedder/          # Embedding service to convert raw text to embeddings
├── docker/                # Dockerfiles for each service
├── tests/                 # Contains unit test cases
├── docker-compose.yaml    # Service orchestration
└── pyproject.toml         # Workspace configuration
```

Each service follows a consistent structure:

```bash
apis/{service}/
├── src/
│   ├── {service}/
│   │   ├── app.py         # FastAPI application
│   │   ├── routes/        # API endpoints
│   │   ├── handlers.py    # Business logic
│   │   ├── serialization.py  # Pydantic models
│   │   └── settings.py    # Configuration
│   └── tests/             # Unit & Integration tests
└── pyproject.toml         # Service dependencies
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- uv package manager (python versions and environments management)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd granicus-case-study
```
2. Setup environment variables
- Rename all the `.env.example` files to `.env`. *You can refer to the filepath mentioned in the `env_file` field in the docker-compose.yaml to locate the `.env` files.
- You only have to set your `LLM_API_KEY`. **Note:** For now, we only support OpenAI models, so please key in the openai API key. You have to set this up in both the `.env` files.

1. Start all services with Docker Compose:

```bash
docker-compose up --build
```

This will start:

- Embedder API at http://localhost:8001
- Docparser API at http://localhost:8002
- Chatbot API at http://localhost:8003
- Weaviate vector database at http://localhost:8080

1. Verify services are running:

```bash
curl http://localhost:8000/health
```

### Interact with the services

I didn't get time to build a proper UI for this. So as of now the best way to interact with the API endpoints is the FastAPI docs.

**Upload docs in docparser**
1. First upload docs via the the `/ingest` endpoint in docparser service. You can upload multiple documents at the same time.
2. Currently, we only support uploading (`.txt`, `.md` and `.csv` docs).
3. The `/ingest` endpoint returns a `job_id`. You can get the status of a job by passing this id to the `/ingest/status/{job_id}` endpoint.


**Chat with the RAG chatbot**
1. Now you can pass your query to the `/chat` endpoint in chatbot service. **Note:** *this endpoint takes some time to execute due to the reranker component. My understanding is that once we deploy this system on a GPU machine, the reranker component will execute much faster.*
2. The response will be a JSON response and will contain the answer along with the source chunks that were used by the LLM to answer the query.

### API Documentation

Interactive API documentation is available at:

- Embedder API: http://localhost:8001/docs
- Docparser API: http://localhost:8002/docs
- Chatbot API: http://localhost:8003/docs

## Testing

Run all unit tests from the project root:

```bash
# All tests
uv run pytest

# All test with coverage
uv run pytest --cov ./src/
```

