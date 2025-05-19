"""
GenAI Document Ingestion API Main Application

This module initializes the FastAPI application.
"""
import os
import sys
from typing import Callable, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .utils.logger import setup_logger
from .routes import query, ingestion, admin
from exceptions import AWSRAGError, ErrorCode

# Package handling for imports
if __name__ == "__main__" and __package__ is None:
    file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, file_path)
    __package__ = "app"

# Initialize logger
logger = setup_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

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

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS with more restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    max_age=3600,
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except AWSRAGError as e:
        logger.error(f"AWS RAG Error: {str(e)}", extra=e.context)
        return JSONResponse(
            status_code=500,  # Default to 500 for AWS RAG errors
            content={
                "error": str(e),
                "error_code": e.context.get("error_code", "UNKNOWN_ERROR"),
                "details": e.context
            }
        )
    except HTTPException as e:
        logger.error(f"HTTP Error: {str(e)}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": str(e),
                "error_code": "HTTP_ERROR",
                "details": {"message": e.detail}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "UNKNOWN_ERROR",
                "details": {"message": str(e)}
            }
        )

# Request validation middleware
@app.middleware("http")
async def request_validation_middleware(request: Request, call_next: Callable) -> Response:
    # Add request validation logic here
    # For example, validate content type, required headers, etc.
    return await call_next(request)

# Include routers with versioning
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])

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