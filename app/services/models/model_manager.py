"""
Model management service for multiple LLM providers.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Set, Union, Any
import json
import os
import time
import functools
import uuid
from ...services.storage.dynamodb_service import DynamoDBService
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelType(str, Enum):
    """Enum for supported model types."""
    RAG = "rag"
    BEDROCK_BASE = "bedrock_base"
    BEDROCK_FINETUNED = "bedrock_finetuned"
    TITAN = "titan"
    DEEPSEEK = "deepseek"
    LLAMA = "llama"

class ModelConfig(BaseModel):
    """Configuration for a model."""
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    enabled: bool = True
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelArtifact(BaseModel):
    """Model artifact information."""
    model_id: str
    version: str
    location: str
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    format: Optional[str] = None
    framework: Optional[str] = None
    created_at: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelMetric(BaseModel):
    """Model performance metrics."""
    model_id: str
    metric_id: str
    name: str
    value: Union[float, int, str, Dict[str, Any]]
    type: str  # accuracy, latency, throughput, etc.
    timestamp: Optional[int] = None
    dataset: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelApprovalStatus(str, Enum):
    """Model approval workflow statuses."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_REVIEW = "in_review"

class ModelManager:
    """
    Manager for LLM models across multiple providers with registry capabilities.
    """
    
    def __init__(self, db_service: Optional[DynamoDBService] = None):
        """
        Initialize the model manager with storage.
        
        Args:
            db_service: DynamoDB service for persistence
        """
        self.db = db_service or DynamoDBService()
        self._provider_cache = {}
        
    def register_model(self, model_id: str, config: ModelConfig) -> bool:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            config: Model configuration
            
        Returns:
            bool: True if registration was successful
        """
        # Convert to dict
        model_data = config.dict()
        
        # Add timestamps
        if not model_data.get('created_at'):
            model_data['created_at'] = int(time.time())
        
        model_data['updated_at'] = int(time.time())
        
        # Save to database
        success = self.db.save_model(model_id, model_data)
        if success:
            logger.info(f"Registered model {model_id} of type {config.type}")
        else:
            logger.error(f"Failed to register model {model_id}")
        
        return success
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a model configuration by ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Optional[ModelConfig]: Model configuration if found
        """
        model_data = self.db.get_model(model_id)
        if not model_data:
            return None
        
        try:
            # Convert to ModelConfig
            return ModelConfig(**model_data)
        except Exception as e:
            logger.error(f"Error parsing model data for {model_id}: {str(e)}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of model data
        """
        models = self.db.list_models()
        return models
        
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a model configuration.
        
        Args:
            model_id: Unique identifier for the model
            updates: Dict of fields to update
            
        Returns:
            bool: True if the update was successful
        """
        return self.db.update_model(model_id, updates)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            bool: True if deletion was successful
        """
        return self.db.delete_model(model_id)
    
    def register_model_artifact(self, model_id: str, 
                             artifact: ModelArtifact) -> bool:
        """
        Register a model artifact.
        
        Args:
            model_id: Model identifier
            artifact: Artifact information
            
        Returns:
            bool: True if registration was successful
        """
        # Convert to dict
        artifact_data = artifact.dict()
        
        # Save to database
        return self.db.save_model_artifact(model_id, artifact.version, artifact_data)
    
    def get_model_artifacts(self, model_id: str) -> List[ModelArtifact]:
        """
        Get all artifacts for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List[ModelArtifact]: List of model artifacts
        """
        artifacts_data = self.db.get_model_artifacts(model_id)
        artifacts = []
        
        for artifact_data in artifacts_data:
            try:
                artifacts.append(ModelArtifact(**artifact_data))
            except Exception as e:
                logger.error(f"Error parsing artifact data: {str(e)}")
                
        return artifacts
    
    def record_model_metric(self, model_id: str, metric: ModelMetric) -> bool:
        """
        Record a metric for a model.
        
        Args:
            model_id: Model identifier
            metric: Performance metric information
            
        Returns:
            bool: True if recording was successful
        """
        # Convert to dict
        metric_data = metric.dict()
        
        # Generate ID if not provided
        metric_id = metric.metric_id or f"metric_{uuid.uuid4()}"
        
        # Save to database
        return self.db.save_model_metric(model_id, metric_id, metric_data)
    
    def get_model_metrics(self, model_id: str) -> List[ModelMetric]:
        """
        Get all metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List[ModelMetric]: List of model metrics
        """
        metrics_data = self.db.get_model_metrics(model_id)
        metrics = []
        
        for metric_data in metrics_data:
            try:
                metrics.append(ModelMetric(**metric_data))
            except Exception as e:
                logger.error(f"Error parsing metric data: {str(e)}")
                
        return metrics
    
    def update_model_approval_status(self, model_id: str, 
                                 status: ModelApprovalStatus, 
                                 reviewer: str = None, 
                                 comments: str = None) -> bool:
        """
        Update the approval status of a model.
        
        Args:
            model_id: Model identifier
            status: New approval status
            reviewer: Name of the reviewer
            comments: Review comments
            
        Returns:
            bool: True if update was successful
        """
        updates = {
            'approval_status': status,
            'approval_timestamp': int(time.time())
        }
        
        if reviewer:
            updates['approval_reviewer'] = reviewer
            
        if comments:
            updates['approval_comments'] = comments
            
        return self.db.update_model(model_id, updates)
    
    def compare_models(self, model_ids: List[str], metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_ids: List of model identifiers to compare
            metrics: List of metric names to compare (or all if None)
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        results = {}
        
        for model_id in model_ids:
            model = self.get_model(model_id)
            if not model:
                continue
                
            metrics_list = self.get_model_metrics(model_id)
            
            # Filter metrics if specific ones requested
            if metrics:
                metrics_list = [m for m in metrics_list if m.name in metrics]
                
            # Add to results
            results[model_id] = {
                'model': model.dict(),
                'metrics': [m.dict() for m in metrics_list]
            }
            
        return results 