from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import setup_logger
from app.routes import query

# Initialize logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GenAI Document Ingestion API",
    description="API for document ingestion and retrieval using RAG",
    version="1.0.0"
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

@app.get("/")
async def root():
    return {"message": "Welcome to GenAI Document Ingestion API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 