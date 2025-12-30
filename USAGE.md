# Offline RAG Service Usage Guide

## Prerequisites
1.  **Docker** installed (for containerized run) OR **Python 3.10+** (for local run).
2.  **LLM Model**: Download a GGUF model (e.g., `llama-2-7b-chat.Q4_K_M.gguf` or `mistral-7b-instruct-v0.1.Q4_K_M.gguf`).
    - Place it in the `models/` directory.
    - Update `config.yaml` if the filename differs.

## Running with Docker (Recommended)
1.  Place your PDF/Word documents in `source_documents/`.
2.  Place your model in `models/`.
3.  Run:
    ```bash
    docker-compose up --build
    ```
4.  Access API Docs at `http://localhost:8000/docs`.

## Running Locally
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have issues with llama-cpp-python, refer to their installation guide for pre-built wheels.*
2.  Run the API:
    ```bash
    uvicorn api:app --reload
    ```

## API Usage

### 1. Ingest Documents
Trigger this after adding new files to `source_documents/`.
- **POST** `/ingest`
- Response: `{"status": "Ingestion complete"}`

### 2. Upload Document via API
- **POST** `/upload` (multipart/form-data)

### 3. Query
- **POST** `/query`
- Body: `{"query": "What is the policy on remote work?"}`
- Response:
  ```json
  {
    "answer": "The policy states...",
    "sources": ["source_documents/policy.pdf (Page 5)"]
  }
  ```
