"""
Admin routes for the application.
"""
from fastapi import APIRouter, HTTPException, Depends, status, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import time
import os
import uuid
from pathlib import Path
import json
import boto3

from ..services.models.model_manager import ModelManager, ModelType, ModelConfig, ModelArtifact, ModelMetric, ModelApprovalStatus
from ..services.storage.dynamodb_service import DynamoDBService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize services
model_manager = ModelManager()
s3 = boto3.client('s3')
ARTIFACT_BUCKET = os.environ.get('ARTIFACT_BUCKET', 'genai-model-artifacts')

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/models", response_class=HTMLResponse)
async def models_dashboard(request: Request):
    """
    Admin dashboard for model management.
    """
    models = model_manager.list_models()
    return templates.TemplateResponse(
        "admin/models.html", 
        {
            "request": request, 
            "models": models,
            "model_types": [t.value for t in ModelType]
        }
    )

@router.get("/models/{model_id}", response_class=HTMLResponse)
async def model_details(request: Request, model_id: str):
    """
    Model details page.
    """
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
        # Check if model with this ID already exists
        existing_model = model_manager.get_model(model_id)
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with ID {model_id} already exists"
            )
        
        # Create model config
        config = ModelConfig(
            name=name,
            type=ModelType(model_type),
            endpoint=endpoint,
            description=description,
            enabled=enabled,
            parameters={},
            metadata={}
        )
        
        # Register the model
        model_manager.register_model(model_id, config)
        
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
    updates = {}
    
    if name is not None:
        updates["name"] = name
        
    if endpoint is not None:
        updates["endpoint"] = endpoint
        
    if description is not None:
        updates["description"] = description
        
    if enabled is not None:
        updates["enabled"] = enabled
    
    if approval_status is not None:
        updates["approval_status"] = approval_status
        updates["approval_timestamp"] = int(time.time())
        
        if approval_comments:
            updates["approval_comments"] = approval_comments
    
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

@router.post("/models/{model_id}/delete")
async def delete_model_form(model_id: str):
    """
    Delete a model.
    """
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
        
        # Generate storage path
        artifact_key = f"{model_id}/{version}/{artifact_file.filename}"
        
        # Upload to S3
        contents = await artifact_file.read()
        s3.put_object(
            Bucket=ARTIFACT_BUCKET,
            Key=artifact_key,
            Body=contents
        )
        
        # Get file size
        size_bytes = len(contents)
        
        # Register artifact
        artifact = ModelArtifact(
            model_id=model_id,
            version=version,
            location=f"s3://{ARTIFACT_BUCKET}/{artifact_key}",
            size_bytes=size_bytes,
            format=os.path.splitext(artifact_file.filename)[1].lstrip('.'),
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
    Record a new metric for a model.
    """
    try:
        # Check if model exists
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Try to convert value to numeric if possible
        try:
            if '.' in value:
                metric_value = float(value)
            else:
                metric_value = int(value)
        except ValueError:
            metric_value = value
        
        # Create metric
        metric = ModelMetric(
            model_id=model_id,
            metric_id=f"metric_{uuid.uuid4()}",
            name=name,
            value=metric_value,
            type=metric_type,
            dataset=dataset,
            description=description,
            timestamp=int(time.time())
        )
        
        model_manager.record_model_metric(model_id, metric)
        
        return RedirectResponse(
            url=f"/admin/models/{model_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
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
        
        # Get comparison
        comparison = model_manager.compare_models(model_id_list)
        
        return templates.TemplateResponse(
            "admin/compare_models.html", 
            {
                "request": request, 
                "comparison": comparison,
                "model_ids": model_id_list
            }
        )
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}"
        ) 