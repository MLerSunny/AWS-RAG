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