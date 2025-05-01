"""
API routes for the GenAI document ingestion service.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, status
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, validator
import uuid
import json

from ..services.models.model_manager import ModelManager, ModelType, ModelConfig
from ..services.rlhf.feedback_collector import FeedbackCollector, FeedbackEntry
from ..services.quality.hallucination_detector import HallucinationDetector
from ..services.opensearch_client import OpenSearchClient
from ..services.bedrock_llm import BedrockLLM
from ..services.chunker import Chunker
from ..services.embedder import Embedder
from ..services.experiment.ab_testing import ABTestingManager, AllocationStrategy
from ..services.finetune.bedrock_finetune import BedrockFineTuner, FineTuneConfig, FineTuneStatus
from ..services.connectors.sharepoint_connector import SharePointConnector
from ..utils.logger import setup_logger
from ..config import settings

# Import ServiceNowConnector conditionally
try:
    from ..services.connectors.servicenow_connector import ServiceNowConnector
    has_servicenow = True
except ImportError:
    has_servicenow = False

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
ab_testing_manager = ABTestingManager()
bedrock_finetune = BedrockFineTuner()
sharepoint = SharePointConnector()

# Initialize ServiceNow connector conditionally
if has_servicenow:
    try:
        servicenow = ServiceNowConnector()
    except Exception as e:
        logger.error(f"Error initializing ServiceNow connector: {str(e)}")
        servicenow = None
else:
    logger.warning("ServiceNow connector is not available - pysnow package missing")
    servicenow = None

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
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 50:
            raise ValueError('top_k must be between 1 and 50')
        return v

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
    query: str = ""
    model_id: str = ""
    response_text: str = ""

class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None

class ModelsResponse(BaseModel):
    success: bool = True
    models: List[ModelResponse] = []

class ExperimentRequest(BaseModel):
    name: str
    variants: List[Dict[str, Any]]
    description: Optional[str] = None
    allocation_strategy: str = "equal"

class ExperimentVariantRequest(BaseModel):
    experiment_id: str
    user_id: str

class FineTuneRequest(BaseModel):
    job_name: str
    base_model_id: str
    training_data_path: str
    validation_data_path: Optional[str] = None
    output_data_path: str
    custom_model_name: Optional[str] = None
    description: Optional[str] = None
    hyperparameters: Dict[str, Any] = {}

class SharePointListRequest(BaseModel):
    library_name: str
    folder_path: Optional[str] = ""

class ServiceNowSearchRequest(BaseModel):
    search_term: str
    limit: int = 20

# Original routes
@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest):
    """
    Query the system with a question.
    Supports both RAG and direct LLM queries depending on the model_id.
    """
    # Get the model configuration
    model_config = model_manager.get_model(request.model_id)
    if not model_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {request.model_id} not found"
        )
    
    # Generate a response ID
    response_id = str(uuid.uuid4())
    
    try:
        # Process query based on model type
        if model_config.type == ModelType.RAG:
            # RAG query
            return await process_rag_query(request, response_id, model_config)
        
        elif model_config.type in [ModelType.BEDROCK_BASE, ModelType.BEDROCK_FINETUNED]:
            # Direct LLM query
            return await process_llm_query(request, response_id, model_config)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Unsupported model type: {model_config.type}"
            )
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error processing query: {str(e)}"
        )

async def process_rag_query(request: QueryRequest, response_id: str, model_config: ModelConfig) -> Dict[str, Any]:
    """Process a RAG query with semantic search and hallucination detection."""
    try:
        # Generate embeddings for the query
        query_embedding = embedder.get_embeddings(request.query)
        
        # Search for relevant documents using the embedding
        search_results = opensearch.search(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        if not search_results:
            return {
                "response_id": response_id,
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "confidence": 1.0,
                "model_id": request.model_id,
                "success": True
            }
        
        # Extract sources and build context
        sources = []
        context_texts = []
        
        for hit in search_results:
            source = hit.get("_source", {})
            text = source.get("content", "")
            metadata = source.get("metadata", {})
            filename = metadata.get("filename", "Unknown source")
            
            context_texts.append(text)
            sources.append(filename)
        
        # Join context with appropriate separators
        context = "\n\n".join(context_texts)
        
        # Get answer from Bedrock
        answer = bedrock_llm.generate_response(request.query, context)
        
        # Check for hallucinations
        hallucination_check = hallucination_detector.check_factual_consistency(
            answer, context_texts
        )
        
        # Enhance response if needed
        if hallucination_check["is_hallucination"]:
            answer = hallucination_detector.enhance_response(
                answer, context_texts
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
        raise

async def process_llm_query(request: QueryRequest, response_id: str, model_config: ModelConfig) -> Dict[str, Any]:
    """Process a direct LLM query without retrieval."""
    try:
        # Apply model-specific parameters with user temperature
        params = model_config.parameters.copy()
        params['temperature'] = request.temperature
        
        # Build system prompt
        system_prompt = "You are a helpful AI assistant. Answer the user's question to the best of your ability."
        
        # Get answer from Bedrock
        answer = bedrock_llm.generate_response(
            request.query, 
            system_prompt
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
        raise

@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback for a response to enable RLHF.
    """
    if not request.response_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="response_id is required"
        )
    
    try:
        # Store the feedback using background task
        background_tasks.add_task(
            feedback_collector.add_feedback,
            FeedbackEntry(
                response_id=request.response_id,
                user_id=request.user_id or "anonymous",
                query=request.query or "",
                model_id=request.model_id or "unknown",
                response_text=request.response_text or "",
                is_helpful=request.is_helpful,
                feedback_text=request.feedback or ""
            )
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )

@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Ingest a document into the system.
    """
    try:
        # Validate metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata format. Must be valid JSON."
                )
        
        # Queue document processing in background
        background_tasks.add_task(
            process_document,
            file,
            document_id,
            metadata_dict
        )
        
        return {
            "success": True, 
            "message": "Document ingestion queued successfully",
            "document_id": document_id or str(uuid.uuid4())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )

async def process_document(file: UploadFile, document_id: Optional[str], metadata: Dict):
    """Background task to process and index a document."""
    # Implementation placeholder
    logger.info(f"Processing document {file.filename}, ID: {document_id}")

# A/B testing routes
@router.post("/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment(request: ExperimentRequest):
    """
    Create a new A/B testing experiment.
    """
    try:
        experiment = ab_testing_manager.create_experiment(
            name=request.name,
            variants=request.variants,
            description=request.description,
            allocation_strategy=AllocationStrategy(request.allocation_strategy)
        )
        
        return {
            "success": True,
            "experiment_id": experiment.id,
            "name": experiment.name
        }
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating experiment: {str(e)}"
        )

@router.post("/experiments/{experiment_id}/start", status_code=status.HTTP_200_OK)
async def start_experiment(experiment_id: str, duration_days: Optional[int] = None):
    """
    Start an A/B testing experiment.
    """
    try:
        success = ab_testing_manager.start_experiment(experiment_id, duration_days)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
            
        return {"success": True, "experiment_id": experiment_id, "status": "active"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting experiment: {str(e)}"
        )

@router.post("/experiments/{experiment_id}/stop", status_code=status.HTTP_200_OK)
async def stop_experiment(experiment_id: str):
    """
    Stop an A/B testing experiment.
    """
    try:
        success = ab_testing_manager.stop_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
            
        return {"success": True, "experiment_id": experiment_id, "status": "completed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping experiment: {str(e)}"
        )

@router.get("/experiments", status_code=status.HTTP_200_OK)
async def list_experiments():
    """
    List all A/B testing experiments.
    """
    try:
        experiments = ab_testing_manager.list_experiments()
        return {"success": True, "experiments": experiments}
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing experiments: {str(e)}"
        )

@router.get("/experiments/{experiment_id}", status_code=status.HTTP_200_OK)
async def get_experiment(experiment_id: str):
    """
    Get details of an A/B testing experiment.
    """
    try:
        experiment = ab_testing_manager.get_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
            
        return {"success": True, "experiment": experiment.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting experiment: {str(e)}"
        )

@router.get("/experiments/{experiment_id}/stats", status_code=status.HTTP_200_OK)
async def get_experiment_stats(experiment_id: str):
    """
    Get statistics for an A/B testing experiment.
    """
    try:
        stats = ab_testing_manager.get_experiment_stats(experiment_id)
        
        if "error" in stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=stats["error"]
            )
            
        return {"success": True, "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving experiment statistics: {str(e)}"
        )

@router.post("/experiments/variant", status_code=status.HTTP_200_OK)
async def get_experiment_variant(request: ExperimentVariantRequest):
    """
    Get a variant for a user in an experiment.
    """
    try:
        variant = ab_testing_manager.select_variant(
            request.experiment_id,
            request.user_id
        )
        
        if not variant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or not active"
            )
            
        return {"success": True, "variant": variant}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment variant: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting experiment variant: {str(e)}"
        )

# Fine-tuning routes
@router.get("/finetune/models", status_code=status.HTTP_200_OK)
async def list_finetune_models():
    """
    List models that support fine-tuning.
    """
    try:
        models = bedrock_finetune.get_supported_models()
        return {"success": True, "models": models}
    except Exception as e:
        logger.error(f"Error listing fine-tunable models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing fine-tunable models: {str(e)}"
        )

@router.post("/finetune/jobs", status_code=status.HTTP_201_CREATED)
async def create_finetune_job(request: FineTuneRequest):
    """
    Create a fine-tuning job.
    """
    try:
        config = FineTuneConfig(
            job_name=request.job_name,
            base_model_id=request.base_model_id,
            training_data_path=request.training_data_path,
            validation_data_path=request.validation_data_path,
            output_data_path=request.output_data_path,
            custom_model_name=request.custom_model_name,
            description=request.description,
            hyperparameters=request.hyperparameters
        )
        
        job = bedrock_finetune.create_fine_tune_job(config)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create fine-tuning job"
            )
            
        return {"success": True, "job": job.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating fine-tuning job: {str(e)}"
        )

@router.get("/finetune/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def get_finetune_job(job_id: str):
    """
    Get status of a fine-tuning job.
    """
    try:
        job = bedrock_finetune.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Fine-tuning job {job_id} not found"
            )
            
        return {"success": True, "job": job.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting fine-tuning job: {str(e)}"
        )

@router.post("/finetune/jobs/{job_id}/stop", status_code=status.HTTP_200_OK)
async def stop_finetune_job(job_id: str):
    """
    Stop a fine-tuning job.
    """
    try:
        success = bedrock_finetune.stop_fine_tune_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stop fine-tuning job {job_id}"
            )
            
        return {"success": True, "job_id": job_id, "status": "stopping"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping fine-tuning job: {str(e)}"
        )

@router.delete("/finetune/models/{model_id}", status_code=status.HTTP_200_OK)
async def delete_finetuned_model(model_id: str):
    """
    Delete a fine-tuned model.
    """
    try:
        success = bedrock_finetune.delete_custom_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete custom model {model_id}"
            )
            
        return {"success": True, "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting custom model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting custom model: {str(e)}"
        )

# Data connector routes
@router.get("/connectors/sharepoint/libraries", status_code=status.HTTP_200_OK)
async def list_sharepoint_libraries():
    """
    List SharePoint document libraries.
    """
    try:
        if not sharepoint.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SharePoint connector not configured"
            )
            
        libraries = sharepoint.list_libraries()
        return {"success": True, "libraries": libraries}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing SharePoint libraries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing SharePoint libraries: {str(e)}"
        )

@router.post("/connectors/sharepoint/documents", status_code=status.HTTP_200_OK)
async def list_sharepoint_documents(request: SharePointListRequest):
    """
    List documents in a SharePoint library/folder.
    """
    try:
        if not sharepoint.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SharePoint connector not configured"
            )
            
        documents = sharepoint.list_documents(
            request.library_name, 
            request.folder_path or ""  # Ensure we pass a string, not None
        )
        return {"success": True, "documents": documents}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing SharePoint documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing SharePoint documents: {str(e)}"
        )

@router.get("/connectors/servicenow/incidents", status_code=status.HTTP_200_OK)
async def list_servicenow_incidents(limit: int = 100):
    """List incidents from ServiceNow."""
    if servicenow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow connector is not available. The pysnow package might be missing."
        )
        
    if not servicenow.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow client is not initialized. Check your credentials."
        )
        
    try:
        incidents = servicenow.get_incidents(limit=limit)
        return {
            "success": True,
            "incidents": incidents,
            "count": len(incidents)
        }
    except Exception as e:
        logger.error(f"Error retrieving ServiceNow incidents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ServiceNow incidents: {str(e)}"
        )

@router.get("/connectors/servicenow/knowledge", status_code=status.HTTP_200_OK)
async def list_servicenow_knowledge(limit: int = 100):
    """List knowledge articles from ServiceNow."""
    if servicenow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow connector is not available. The pysnow package might be missing."
        )
        
    if not servicenow.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow client is not initialized. Check your credentials."
        )
        
    try:
        articles = servicenow.get_knowledge_articles(limit=limit)
        return {
            "success": True,
            "articles": articles,
            "count": len(articles)
        }
    except Exception as e:
        logger.error(f"Error retrieving ServiceNow knowledge articles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ServiceNow knowledge articles: {str(e)}"
        )

@router.post("/connectors/servicenow/search", status_code=status.HTTP_200_OK)
async def search_servicenow_knowledge(request: ServiceNowSearchRequest):
    """Search knowledge articles in ServiceNow."""
    if servicenow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow connector is not available. The pysnow package might be missing."
        )
        
    if not servicenow.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ServiceNow client is not initialized. Check your credentials."
        )
        
    try:
        results = servicenow.search_knowledge_base(request.search_term, request.limit)
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching ServiceNow knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching ServiceNow knowledge base: {str(e)}"
        ) 