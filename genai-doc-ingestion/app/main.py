"""
GenAI Document Ingestion API Main Application

This module initializes the FastAPI application.
"""
import os
import sys

# Package handling for imports
if __name__ == "__main__" and __package__ is None:
    file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, file_path)
    __package__ = "app"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .utils.logger import setup_logger
from .routes import query, ingestion

# Initialize logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GenAI Document Ingestion API",
    description="""
    API for document ingestion and retrieval using RAG (Retrieval Augmented Generation) with AWS Bedrock and OpenSearch.
    
    ## Features
    - **Document Ingestion**: Upload documents (PDF, DOCX, TXT) for processing and indexing
    - **Batch Processing**: Upload multiple files at once for efficient processing
    - **Semantic Search**: Query documents using natural language
    - **RAG**: Retrieval Augmented Generation for better AI responses using your data
    
    This API provides endpoints for adding documents to the knowledge base and querying information using natural language.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["ingestion"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to GenAI Document Ingestion API",
        "documentation": "/api/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 