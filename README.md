# RAG chatbot system

A scalable production-ready RAG chatbot system that can answer questions about Granicus products and services. The system must handle diverse document formats, provide accurate information, and maintain strict adherence to the provided knowledge base.

## Overview

## Core components

- **Document Parser**: Ingests documents in a variety of formats and chunks them using custom chunking logic.
- **Embedder**: Create vector embeddings of the input texts using sentence transformers.
- **Vector Database**: Weaviate vector database for high-performance vector similarity search
-- **Retriever**:  Retrieves documents from vector database based on distance from the query vector.
- **Reranker**: Reranks retrieved documents using Jina reranker model. 
- **Chatbot**: This is where all the components come together and make the RAG pipeline

## Architecture

The system consists of four main microservices:

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

1. Start all services with Docker Compose:

```bash
docker-compose up --build
```

This will start:

- Chatbot API at http://localhost:8003
- Embedder API at http://localhost:8001
- Docparser API at http://localhost:8002
- Weaviate vector database at http://localhost:8080

1. Verify services are running:

```bash
curl http://localhost:8000/health
```

### Local Development

For development without Docker:

1. Install dependencies:

```bash
# Install uv if not already installed
pip install uv

# Install workspace dependencies
uv sync --all-packages
```

2. Run individual services from the project root:

```bash
# Embedder service
uv --directory=src/apis/embedder/src/embedder run --package=embedder fastapi dev app.py --host=0.0.0.0 --port=8001

# Docparser service
uv --directory=src/apis/docparser/src/docparser run --package=docparser fastapi dev app.py --host=0.0.0.0 --port=8002

# Search service
uv --directory=src/apis/chatbot/src/chatbot run --package=chatbot fastapi dev app.py --host=0.0.0.0 --port=8003
```

### API Documentation

Interactive API documentation is available at:

- Embedder API: http://localhost:8001/docs
- Docparser API: http://localhost:8002/docs
- Chatbot API: http://localhost:8003/docs

## [NEED TO VERIFY] Testing

Run all unit tests:

```bash
# All tests
uv run pytest

# Specific service
cd apis/search
uv run pytest src/tests/
```

## [NEED TO VERIFY] Configuration

Each service can be configured via environment variables. See individual service `settings.py` files for available options.

Common configuration options:
- `EMBEDDER_URL`: URL of the embedder service
- `RERANKER_URL`: URL of the reranker service
- `VECTORDB_URL`: Vector database connection URL
