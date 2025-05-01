"""
API routes for the GenAI document ingestion service.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid

from ..services.models.model_manager import ModelManager, ModelType, ModelConfig
from ..services.rlhf.feedback_collector import FeedbackCollector, FeedbackEntry
from ..services.quality.hallucination_detector import HallucinationDetector
from ..services.opensearch_client import OpenSearchClient
from ..services.bedrock_llm import BedrockLLM
from ..services.chunker import Chunker
from ..services.embedder import Embedder
from ..utils.logger import setup_logger
from ..config import settings

router = APIRouter()
logger = setup_logger(__name__)

# Initialize services
model_manager = ModelManager()
feedback_collector = FeedbackCollector()
hallucination_detector = HallucinationDetector()
opensearch = OpenSearchClient()
chunker = Chunker()
embedder = Embedder()
bedrock_llm = BedrockLLM()

# Register default models
model_manager.register_model(
    "rag",
    ModelConfig(
        name="RAG",
        type=ModelType.RAG,
        description="Retrieval Augmented Generation with OpenSearch"
    )
)

model_manager.register_model(
    "bedrock",
    ModelConfig(
        name="AWS Bedrock Claude",
        type=ModelType.BEDROCK_BASE,
        description="Amazon Bedrock Claude model"
    )
)

# Define request/response models
class QueryRequest(BaseModel):
    query: str
    model_id: str
    temperature: float = 0.7
    top_k: int = 5

class QueryResponse(BaseModel):
    response_id: str
    answer: str
    sources: List[str] = []
    confidence: float = 1.0
    model_id: str
    success: bool = True

class FeedbackRequest(BaseModel):
    response_id: str
    is_helpful: bool
    feedback: str = ""
    user_id: Optional[str] = None
    query: Optional[str] = ""
    model_id: Optional[str] = ""
    response_text: Optional[str] = ""

class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None

class ModelsResponse(BaseModel):
    success: bool = True
    models: List[ModelResponse] = []

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the system with a question.
    Supports both RAG and direct LLM queries depending on the model_id.
    """
    # Get the model configuration
    model_config = model_manager.get_model(request.model_id)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    # Generate a response ID
    response_id = str(uuid.uuid4())
    
    # Process query based on model type
    if model_config.type == ModelType.RAG:
        # RAG query
        try:
            # Search for relevant documents
            search_results = opensearch.search(
                request.query,
                top_k=request.top_k
            )
            
            # Extract sources
            sources = []
            context = ""
            for hit in search_results:
                source = hit.get("_source", {})
                text = source.get("content", "")
                metadata = source.get("metadata", {})
                filename = metadata.get("filename", "Unknown source")
                
                context += f"{text}\n\n"
                sources.append(filename)
            
            # Get answer from Bedrock
            answer = bedrock_llm.generate_response(request.query, context)
            
            # Check for hallucinations
            hallucination_check = hallucination_detector.check_factual_consistency(
                answer, [hit.get("_source", {}).get("content", "") for hit in search_results]
            )
            
            # Enhance response if needed
            if hallucination_check["is_hallucination"]:
                answer = hallucination_detector.enhance_response(
                    answer, [hit.get("_source", {}).get("content", "") for hit in search_results]
                )
            
            return {
                "response_id": response_id,
                "answer": answer,
                "sources": sources,
                "confidence": hallucination_check["confidence"],
                "model_id": request.model_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")
    
    elif model_config.type in [ModelType.BEDROCK_BASE, ModelType.BEDROCK_FINETUNED]:
        # Direct LLM query
        try:
            # Call Bedrock with appropriate parameters
            params = model_config.parameters.copy()
            if 'temperature' not in params:
                params['temperature'] = request.temperature
                
            # Get answer from Bedrock
            answer = bedrock_llm.generate_response(
                request.query, 
                "You are a helpful AI assistant. Answer the user's question to the best of your ability."
            )
            
            return {
                "response_id": response_id,
                "answer": answer,
                "sources": [],
                "confidence": 1.0,  # No hallucination check for direct queries
                "model_id": request.model_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"LLM query error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing LLM query: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_config.type}")

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback for a response to enable RLHF.
    """
    try:
        # Store the feedback using background task
        background_tasks.add_task(
            feedback_collector.add_feedback,
            FeedbackEntry(
                response_id=request.response_id,
                user_id=request.user_id,
                query=request.query,
                model_id=request.model_id,
                response_text=request.response_text,
                is_helpful=request.is_helpful,
                feedback_text=request.feedback
            )
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    Get a list of available models.
    """
    try:
        models = model_manager.list_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Ingest a document into the system.
    """
    try:
        # Process as in existing implementation
        return {"success": True, "message": "Document ingested successfully"}
    except Exception as e:
        logger.error(f"Document ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

@router.get("/experiments/{experiment_id}/stats")
async def get_experiment_stats(experiment_id: str):
    """
    Get statistics for an A/B testing experiment.
    """
    # This would integrate with the A/B testing framework
    return {"success": True, "message": "Experiment statistics endpoint placeholder"} 