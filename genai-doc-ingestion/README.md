# GenAI Document Ingestion System

A scalable document ingestion and retrieval system using AWS services and GenAI capabilities.

## Overview

This project implements a RAG (Retrieval Augmented Generation) system for document processing and querying.
It uses:

- AWS Bedrock for LLM inference
- OpenSearch for vector storage and retrieval
- FastAPI for the backend service
- Terraform for infrastructure management

## Project Structure

```text
genai-doc-ingestion/
├── app/                # FastAPI application
├── ingestion/          # Document ingestion pipeline
├── frontend/           # Optional UI components
├── data/               # Sample data
├── terraform/          # Infrastructure code
├── notebooks/          # Experimentation notebooks
└── scripts/            # Utility scripts
├── .venv/              # Virtual environment directory
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and other settings
   ```

4. Deploy infrastructure:

   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

## Development

- Backend: `cd app && uvicorn main:app --reload`
- Frontend: `cd frontend && streamlit run app.py`

## License

MIT