"""
Admin routes for the application.
"""
from fastapi import APIRouter, HTTPException, Depends, status, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import time
import os
import uuid
from pathlib import Path
import json
import boto3
from botocore.exceptions import ClientError

from ..services.models.model_manager import ModelManager, ModelType, ModelConfig, ModelArtifact, ModelMetric, ModelApprovalStatus
from ..services.storage.dynamodb_service import DynamoDBService
from ..utils.logger import setup_logger
from ..utils.validation import validate_filename
from ..utils.cache import Cache
from app.services.finetune.bedrock_finetune import BedrockFineTuner, FineTuneConfig
from app.config import settings

logger = setup_logger(__name__)

# Initialize services
model_manager = ModelManager()
s3 = boto3.client('s3')
ARTIFACT_BUCKET = os.environ.get('ARTIFACT_BUCKET', 'genai-model-artifacts')

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

# Initialize cache
cache = Cache(ttl=3600)  # 1 hour TTL

router = APIRouter(prefix="/admin", tags=["admin"])

class ModelCreateRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    model_type: str
    endpoint: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    
    @validator('model_type')
    def validate_model_type(cls, v):
        try:
            ModelType(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid model type. Must be one of: {[t.value for t in ModelType]}")

class ModelUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    endpoint: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    approval_status: Optional[str] = None
    approval_comments: Optional[str] = None
    
    @validator('approval_status')
    def validate_approval_status(cls, v):
        if v is not None:
            try:
                ModelApprovalStatus(v)
                return v
            except ValueError:
                raise ValueError(f"Invalid approval status. Must be one of: {[s.value for s in ModelApprovalStatus]}")

@router.get("/models", response_class=HTMLResponse)
async def models_dashboard(request: Request):
    """
    Admin dashboard for model management.
    """
    try:
        models = model_manager.list_models()
        return templates.TemplateResponse(
            "admin/models.html", 
            {
                "request": request, 
                "models": models,
                "model_types": [t.value for t in ModelType]
            }
        )
    except Exception as e:
        logger.error(f"Error loading models dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading models dashboard: {str(e)}"
        )

@router.get("/models/{model_id}", response_class=HTMLResponse)
async def model_details(request: Request, model_id: str):
    """
    Model details page.
    """
    try:
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Get artifacts and metrics
        artifacts = model_manager.get_model_artifacts(model_id)
        metrics = model_manager.get_model_metrics(model_id)
        
        return templates.TemplateResponse(
            "admin/model_details.html", 
            {
                "request": request, 
                "model": model,
                "model_id": model_id,
                "artifacts": artifacts,
                "metrics": metrics,
                "model_types": [t.value for t in ModelType],
                "approval_statuses": [s.value for s in ModelApprovalStatus]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model details: {str(e)}"
        )

@router.post("/models/create")
async def create_model_form(
    model_id: str = Form(...),
    name: str = Form(...),
    model_type: str = Form(...),
    endpoint: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    enabled: bool = Form(True)
):
    """
    Create a new model from form submission.
    """
    try:
        # Validate input
        request = ModelCreateRequest(
            model_id=model_id,
            name=name,
            model_type=model_type,
            endpoint=endpoint,
            description=description,
            enabled=enabled
        )
        
        # Check if model with this ID already exists
        existing_model = model_manager.get_model(request.model_id)
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with ID {request.model_id} already exists"
            )
        
        # Create model config
        config = ModelConfig(
            name=request.name,
            type=ModelType(request.model_type),
            endpoint=request.endpoint,
            description=request.description,
            enabled=request.enabled,
            parameters={},
            metadata={}
        )
        
        # Register the model
        model_manager.register_model(request.model_id, config)
        
        return RedirectResponse(
            url=f"/admin/models/{request.model_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating model: {str(e)}"
        )

@router.post("/models/{model_id}/update")
async def update_model_form(
    model_id: str,
    name: Optional[str] = Form(None),
    endpoint: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    enabled: Optional[bool] = Form(None),
    approval_status: Optional[str] = Form(None),
    approval_comments: Optional[str] = Form(None)
):
    """
    Update a model from form submission.
    """
    try:
        # Validate input
        request = ModelUpdateRequest(
            name=name,
            endpoint=endpoint,
            description=description,
            enabled=enabled,
            approval_status=approval_status,
            approval_comments=approval_comments
        )
        
        updates = {}
        
        if request.name is not None:
            updates["name"] = request.name
            
        if request.endpoint is not None:
            updates["endpoint"] = request.endpoint
            
        if request.description is not None:
            updates["description"] = request.description
            
        if request.enabled is not None:
            updates["enabled"] = request.enabled
        
        if request.approval_status is not None:
            updates["approval_status"] = request.approval_status
            updates["approval_timestamp"] = int(time.time())
            
            if request.approval_comments:
                updates["approval_comments"] = request.approval_comments
        
        if not updates:
            return RedirectResponse(
                url=f"/admin/models/{model_id}",
                status_code=status.HTTP_303_SEE_OTHER
            )
        
        success = model_manager.update_model(model_id, updates)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        return RedirectResponse(
            url=f"/admin/models/{model_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating model: {str(e)}"
        )

@router.post("/models/{model_id}/delete")
async def delete_model_form(model_id: str):
    """
    Delete a model.
    """
    try:
        success = model_manager.delete_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        return RedirectResponse(
            url="/admin/models",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )

@router.post("/models/{model_id}/upload-artifact")
async def upload_artifact(
    model_id: str,
    version: str = Form(...),
    framework: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    artifact_file: UploadFile = File(...)
):
    """
    Upload a model artifact.
    """
    try:
        # Check if model exists
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Validate filename
        if not artifact_file.filename or not validate_filename(artifact_file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid filename: {artifact_file.filename}"
            )
        
        # Generate storage path
        artifact_key = f"{model_id}/{version}/{artifact_file.filename}"
        
        # Upload to S3
        try:
            contents = await artifact_file.read()
            s3.put_object(
                Bucket=ARTIFACT_BUCKET,
                Key=artifact_key,
                Body=contents
            )
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error uploading artifact: {str(e)}"
            )
        
        # Get file size
        size_bytes = len(contents)
        
        # Get file extension
        _, ext = os.path.splitext(artifact_file.filename)
        file_format = ext.lstrip('.') if ext else ''
        
        # Register artifact
        artifact = ModelArtifact(
            model_id=model_id,
            version=version,
            location=f"s3://{ARTIFACT_BUCKET}/{artifact_key}",
            size_bytes=size_bytes,
            format=file_format,
            framework=framework,
            metadata={
                "filename": artifact_file.filename,
                "description": description
            }
        )
        
        model_manager.register_model_artifact(model_id, artifact)
        
        return RedirectResponse(
            url=f"/admin/models/{model_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading artifact: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading artifact: {str(e)}"
        )

@router.post("/models/{model_id}/record-metric")
async def record_metric(
    model_id: str,
    name: str = Form(...),
    value: str = Form(...),
    metric_type: str = Form(...),
    dataset: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """
    Record a model metric.
    """
    try:
        # Check if model exists
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Parse metric value
        try:
            if metric_type == "float":
                metric_value = float(value)
            elif metric_type == "int":
                metric_value = int(value)
            elif metric_type == "bool":
                metric_value = value.lower() in ("true", "1", "yes")
            else:
                raise ValueError(f"Invalid metric type: {metric_type}")
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric value: {str(e)}"
            )
        
        # Create metric
        metric = ModelMetric(
            model_id=model_id,
            metric_id=f"metric_{uuid.uuid4()}",
            name=name,
            value=str(metric_value),  # Convert to string for storage
            type=metric_type,
            dataset=dataset,
            description=description,
            timestamp=int(time.time())
        )
        
        # Record metric
        model_manager.record_model_metric(model_id, metric)
        
        return RedirectResponse(
            url=f"/admin/models/{model_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording metric: {str(e)}"
        )

@router.get("/models/compare", response_class=HTMLResponse)
async def compare_models_page(request: Request, model_ids: str):
    """
    Compare multiple models.
    """
    try:
        # Parse model IDs
        model_id_list = model_ids.split(',')
        if not model_id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No model IDs provided"
            )
        
        # Get model details
        models = []
        for model_id in model_id_list:
            model = model_manager.get_model(model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_id} not found"
                )
            models.append(model)
        
        # Get metrics for all models
        all_metrics = {}
        for model in models:
            metrics = model_manager.get_model_metrics(model.model_id)
            all_metrics[model.model_id] = metrics
        
        return templates.TemplateResponse(
            "admin/compare_models.html",
            {
                "request": request,
                "models": models,
                "metrics": all_metrics
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}"
        )

@router.post("/finetune/trigger", status_code=201)
async def trigger_finetune(
    job_name: str,
    base_model_id: str,
    custom_model_name: str = None,
    description: str = None
):
    """
    Admin endpoint to trigger a Bedrock fine-tuning job using processed_interactions.jsonl.
    """
    processed_file = "processed_interactions.jsonl"
    s3_bucket = settings.FINETUNE_OUTPUT_BUCKET
    if not s3_bucket:
        raise HTTPException(status_code=400, detail="FINETUNE_OUTPUT_BUCKET is not set in settings.")
    if not job_name or not base_model_id:
        raise HTTPException(status_code=400, detail="job_name and base_model_id are required.")
    s3_key = f"finetune-data/{job_name}.jsonl"
    output_data_path = f"s3://{s3_bucket}/finetune-output/{job_name}/"
    training_data_path = f"s3://{s3_bucket}/{s3_key}"

    bedrock_finetune = BedrockFineTuner()
    # Upload file to S3
    try:
        uploaded = bedrock_finetune.upload_training_file(processed_file, s3_bucket, s3_key)
        if not uploaded:
            raise Exception("Upload to S3 failed")
    except Exception as e:
        logger.error(f"Failed to upload training file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload training file: {str(e)}")

    # Create fine-tune config
    config = FineTuneConfig(
        job_name=job_name,
        base_model_id=base_model_id,
        training_data_path=training_data_path,
        output_data_path=output_data_path,
        custom_model_name=custom_model_name or "",
        description=description or ""
    )
    # Trigger fine-tuning job
    try:
        job = bedrock_finetune.create_fine_tune_job(config)
        return {"success": True, "job": job.to_dict()}
    except Exception as e:
        logger.error(f"Failed to create fine-tuning job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create fine-tuning job: {str(e)}")

@router.get("/finetune/status/{job_id}", status_code=200)
async def get_finetune_status(job_id: str):
    """
    Get the status of a Bedrock fine-tuning job by job_id.
    Sends a notification if the job is completed or failed (placeholder for notification logic).
    """
    bedrock_finetune = BedrockFineTuner()
    try:
        job = bedrock_finetune.get_job_status(job_id)
        status = job.status.value if hasattr(job.status, 'value') else str(job.status)
        # Placeholder: Notification logic
        if status in ["completed", "failed"]:
            # TODO: Integrate with email/Slack notification service here
            logger.info(f"Notification: Fine-tuning job {job_id} status is {status}.")
        return {"success": True, "job": job.to_dict()}
    except Exception as e:
        logger.error(f"Failed to get fine-tuning job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.post("/models/register", status_code=201)
async def register_model(
    model_id: str,
    name: str,
    model_type: str,
    description: str = "",
    endpoint: str = "",
    metadata: Optional[dict] = None
):
    """
    Register a new model (base or fine-tuned) in the model registry.
    """
    try:
        config = ModelConfig(
            name=name,
            type=ModelType(model_type),
            endpoint=endpoint or None,
            description=description or None,
            enabled=True,
            parameters={},
            metadata=metadata if metadata is not None else {}
        )
        model_manager.register_model(model_id, config)
        return {"success": True, "model_id": model_id}
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering model: {str(e)}")

@router.post("/models/{model_id}/activate", status_code=200)
async def activate_model(model_id: str):
    """
    Activate a model for serving (set as active in the registry).
    """
    try:
        updates = {"enabled": True}
        all_models = model_manager.list_models()
        this_model = model_manager.get_model(model_id)
        if not this_model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        for m in all_models:
            m_type = getattr(m, "type", None) if not isinstance(m, dict) else m.get("type")
            m_id = getattr(m, "id", None) if not isinstance(m, dict) else m.get("id")
            if m_type == this_model.type and m_id and m_id != model_id:
                model_manager.update_model(m_id, {"enabled": False})
        model_manager.update_model(model_id, updates)
        return {"success": True, "model_id": model_id, "activated": True}
    except Exception as e:
        logger.error(f"Error activating model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error activating model: {str(e)}")

@router.get("/analytics/usage", status_code=200)
async def analytics_usage():
    """
    Get basic usage stats: total queries and queries per model.
    """
    from app.services.storage.dynamodb_service import DynamoDBService
    db_service = DynamoDBService()
    table_name = "genai_user_interactions"
    table = db_service._get_table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    total_queries = len(items)
    queries_per_model = {}
    for item in items:
        model_id = item.get("model_id", "unknown")
        queries_per_model[model_id] = queries_per_model.get(model_id, 0) + 1
    return {"total_queries": total_queries, "queries_per_model": queries_per_model}

@router.get("/analytics/feedback", status_code=200)
async def analytics_feedback():
    """
    Get feedback stats: helpful/unhelpful counts per model.
    """
    from app.services.storage.dynamodb_service import DynamoDBService
    db_service = DynamoDBService()
    table_name = "genai_user_interactions"
    table = db_service._get_table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    feedback_stats = {}
    for item in items:
        model_id = item.get("model_id", "unknown")
        is_helpful = item.get("is_helpful")
        if model_id not in feedback_stats:
            feedback_stats[model_id] = {"helpful": 0, "unhelpful": 0}
        if is_helpful is True:
            feedback_stats[model_id]["helpful"] += 1
        elif is_helpful is False:
            feedback_stats[model_id]["unhelpful"] += 1
    return {"feedback_stats": feedback_stats}

@router.get("/analytics/variant", status_code=200)
async def analytics_variant():
    """
    Get per-variant stats: queries and feedback per experiment/variant.
    """
    from app.services.storage.dynamodb_service import DynamoDBService
    db_service = DynamoDBService()
    table_name = "genai_user_interactions"
    table = db_service._get_table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    variant_stats = {}
    for item in items:
        experiment_id = item.get("experiment_id")
        variant_id = item.get("variant_id")
        model_id = item.get("model_id", "unknown")
        is_helpful = item.get("is_helpful")
        if experiment_id and variant_id:
            key = f"{experiment_id}:{variant_id}"
            if key not in variant_stats:
                variant_stats[key] = {"experiment_id": experiment_id, "variant_id": variant_id, "model_id": model_id, "queries": 0, "helpful": 0, "unhelpful": 0}
            variant_stats[key]["queries"] += 1
            if is_helpful is True:
                variant_stats[key]["helpful"] += 1
            elif is_helpful is False:
                variant_stats[key]["unhelpful"] += 1
    return {"variant_stats": list(variant_stats.values())}

@router.get("/analytics/timetrends", status_code=200)
async def analytics_timetrends():
    """
    Get time trends: queries per day for the last 30 days.
    """
    from app.services.storage.dynamodb_service import DynamoDBService
    from datetime import datetime, timedelta
    db_service = DynamoDBService()
    table_name = "genai_user_interactions"
    table = db_service._get_table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    today = datetime.utcnow().date()
    trends = {}
    for i in range(30):
        day = today - timedelta(days=i)
        trends[day.isoformat()] = 0
    for item in items:
        ts = item.get("timestamp")
        if ts:
            try:
                dt = datetime.utcfromtimestamp(float(ts))
                day = dt.date().isoformat()
                if day in trends:
                    trends[day] += 1
            except Exception:
                continue
    return {"queries_per_day": trends}

@router.get("/analytics/engagement", status_code=200)
async def analytics_engagement():
    """
    Get user engagement stats: unique users, average queries per user.
    """
    from app.services.storage.dynamodb_service import DynamoDBService
    db_service = DynamoDBService()
    table_name = "genai_user_interactions"
    table = db_service._get_table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    user_counts = {}
    for item in items:
        user_id = item.get("user_id", "anonymous")
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
    unique_users = len(user_counts)
    total_queries = len(items)
    avg_queries_per_user = total_queries / unique_users if unique_users else 0
    return {"unique_users": unique_users, "avg_queries_per_user": avg_queries_per_user}

@router.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """
    Admin analytics dashboard page with visualizations.
    """
    from fastapi import BackgroundTasks
    import httpx
    # Fetch analytics data from internal endpoints
    base_url = str(request.base_url).rstrip("/")
    async with httpx.AsyncClient() as client:
        usage = (await client.get(f"{base_url}/admin/analytics/usage")).json()
        feedback = (await client.get(f"{base_url}/admin/analytics/feedback")).json()
        variant = (await client.get(f"{base_url}/admin/analytics/variant")).json()
        timetrends = (await client.get(f"{base_url}/admin/analytics/timetrends")).json()
        engagement = (await client.get(f"{base_url}/admin/analytics/engagement")).json()
    return templates.TemplateResponse(
        "admin/analytics.html",
        {
            "request": request,
            "usage": usage,
            "feedback": feedback,
            "variant": variant,
            "timetrends": timetrends,
            "engagement": engagement
        }
    ) 