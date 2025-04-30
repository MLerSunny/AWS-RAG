# GenAI Document Ingestion API

A FastAPI-based API for document ingestion and retrieval using RAG (Retrieval Augmented Generation) with AWS Bedrock and OpenSearch.

## Features

- **Document Ingestion**: Upload documents (PDF, DOCX, TXT) for processing and indexing
- **Batch Processing**: Upload multiple files at once for efficient processing
- **Semantic Search**: Query documents using natural language
- **RAG**: Retrieval Augmented Generation for better AI responses using your data

## Prerequisites

- Python 3.8 or higher
- OpenSearch instance (local or remote)
- AWS account with Bedrock access (optional, falls back to local models)

## Quick Start

### Windows

1. Run the batch file: `run.bat`

This will automatically set up the environment and start the application.

### Linux/Mac

1. Make the script executable: `chmod +x run.sh`
2. Run the shell script: `./run.sh`

This will automatically set up the environment and start the application.

### Manual Setup

1. Install Python 3.8+
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file with appropriate settings
6. Start the application:

   ```bash
   python start.py --reload
   ```

## API Endpoints

### Document Ingestion

- **POST** `/api/v1/ingest`: Upload a single document
  - Parameters:
    - `file`: The document file to upload (PDF, DOCX, TXT)
    - `document_id` (optional): Custom ID for the document
    - `metadata` (optional): JSON string with document metadata

- **POST** `/api/v1/batch-ingest`: Upload multiple documents
  - Parameters:
    - `files`: List of document files to upload
    - `metadata` (optional): JSON string with shared metadata

### Document Query

- **POST** `/api/v1/query`: Query for information
  - Request Body:

    ```json
    {
      "query": "What is the capital of France?",
      "top_k": 5,
      "use_local_embeddings": true
    }
    ```

  - Response:

    ```json
    {
      "answer": "The capital of France is Paris.",
      "sources": ["document1.pdf", "document2.txt"],
      "success": true
    }
    ```

## Configuration

Configuration is handled through environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| AWS_REGION | AWS region for Bedrock | us-east-1 |
| AWS_ACCESS_KEY_ID | AWS access key | None |
| AWS_SECRET_ACCESS_KEY | AWS secret key | None |
| OPENSEARCH_HOST | OpenSearch host | localhost |
| OPENSEARCH_PORT | OpenSearch port | 9200 |
| OPENSEARCH_INDEX | OpenSearch index name | documents |
| BEDROCK_MODEL_ID | Bedrock model ID | anthropic.claude-v2 |
| ENVIRONMENT | Environment (development/production) | development |
| DEBUG | Enable debug mode | False |
| CHUNK_SIZE | Document chunk size | 1000 |
| CHUNK_OVERLAP | Chunk overlap size | 200 |

## Development

### Project Structure

- `app/`: Main application code
  - `main.py`: FastAPI application entry point
  - `config.py`: Configuration settings
  - `routes/`: API route definitions
  - `services/`: Service implementations
  - `utils/`: Utility functions
- `start.py`: Application startup script
- `setup.py`: Environment setup script

### Running Tests

```bash
pytest
```

## License

MIT
