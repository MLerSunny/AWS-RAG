from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from app.services.storage.dynamodb_service import DynamoDBService
from app.utils.logger import setup_logger
from app.utils.dynamodb_types import DynamoDBItem, DynamoDBTypeConverter

logger = setup_logger(__name__)

class ModelArtifact(DynamoDBItem['ModelArtifact']):
    """Model for storing model artifact metadata."""
    
    model_id: str = Field(..., description="ID of the parent model")
    version: str = Field(..., description="Version identifier")
    artifact_type: str = Field(..., description="Type of artifact (weights, config, etc.)")
    s3_path: str = Field(..., description="S3 path to the artifact")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    
    @validator('version')
    def validate_version(cls, v):
        """Validate version format (e.g., v1.0.0)."""
        if not v.startswith('v'):
            raise ValueError("Version must start with 'v'")
        try:
            # Split version and validate numbers
            parts = v[1:].split('.')
            if len(parts) != 3:
                raise ValueError
            [int(x) for x in parts]
        except (ValueError, IndexError):
            raise ValueError("Version must be in format vX.Y.Z where X, Y, Z are numbers")
        return v

class ModelVersion(DynamoDBItem['ModelVersion']):
    """Model for storing model version metadata."""
    
    model_id: str = Field(..., description="ID of the parent model")
    version: str = Field(..., description="Version identifier")
    status: str = Field(..., description="Version status (training, active, archived)")
    metrics: Dict[str, float] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    updated_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    
    @validator('status')
    def validate_status(cls, v):
        """Validate version status."""
        valid_statuses = ['training', 'active', 'archived']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

class ModelRegistry:
    """Service for managing model versions and artifacts."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        
    def register_version(self, model_id: str, version: str, hyperparameters: Dict[str, Any]) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            hyperparameters: Model hyperparameters
            
        Returns:
            ModelVersion instance
        """
        version_data = ModelVersion(
            model_id=model_id,
            version=version,
            status='training',
            hyperparameters=hyperparameters
        )
        
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_versions")
            table.put_item(Item=version_data.to_dynamodb())
            return version_data
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            raise
    
    def update_version_status(self, model_id: str, version: str, status: str) -> bool:
        """
        Update the status of a model version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            status: New status
            
        Returns:
            bool: True if update was successful
        """
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_versions")
            table.update_item(
                Key={
                    'model_id': model_id,
                    'version': version
                },
                UpdateExpression="SET #status = :status, updated_at = :updated_at",
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': status,
                    ':updated_at': Decimal(str(datetime.utcnow().timestamp()))
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating model version status: {str(e)}")
            return False
    
    def add_artifact(self, model_id: str, version: str, artifact_type: str, s3_path: str, metadata: Dict[str, Any]) -> ModelArtifact:
        """
        Add an artifact to a model version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            artifact_type: Type of artifact
            s3_path: S3 path to the artifact
            metadata: Additional metadata
            
        Returns:
            ModelArtifact instance
        """
        artifact = ModelArtifact(
            model_id=model_id,
            version=version,
            artifact_type=artifact_type,
            s3_path=s3_path,
            metadata=metadata
        )
        
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_artifacts")
            table.put_item(Item=artifact.to_dynamodb())
            return artifact
        except Exception as e:
            logger.error(f"Error adding model artifact: {str(e)}")
            raise
    
    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """
        Get a model version by ID and version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            
        Returns:
            Optional[ModelVersion]: Model version if found
        """
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_versions")
            response = table.get_item(
                Key={
                    'model_id': model_id,
                    'version': version
                }
            )
            
            if 'Item' in response:
                return ModelVersion.from_dynamodb(response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error getting model version: {str(e)}")
            return None
    
    def get_version_artifacts(self, model_id: str, version: str) -> List[ModelArtifact]:
        """
        Get all artifacts for a model version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            
        Returns:
            List[ModelArtifact]: List of artifacts
        """
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_artifacts")
            response = table.query(
                KeyConditionExpression="model_id = :model_id AND version = :version",
                ExpressionAttributeValues={
                    ':model_id': model_id,
                    ':version': version
                }
            )
            
            return [ModelArtifact.from_dynamodb(item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error getting model artifacts: {str(e)}")
            return []
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """
        List all versions of a model.
        
        Args:
            model_id: ID of the parent model
            
        Returns:
            List[ModelVersion]: List of model versions
        """
        try:
            table = self.db_service._get_table(f"{self.table_prefix}model_versions")
            response = table.query(
                KeyConditionExpression="model_id = :model_id",
                ExpressionAttributeValues={
                    ':model_id': model_id
                }
            )
            
            return [ModelVersion.from_dynamodb(item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            return []
    
    def update_version_metrics(self, model_id: str, version: str, metrics: Dict[str, float]) -> bool:
        """
        Update metrics for a model version.
        
        Args:
            model_id: ID of the parent model
            version: Version identifier
            metrics: New metrics
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Convert float metrics to Decimal
            decimal_metrics = {k: Decimal(str(v)) for k, v in metrics.items()}
            
            table = self.db_service._get_table(f"{self.table_prefix}model_versions")
            table.update_item(
                Key={
                    'model_id': model_id,
                    'version': version
                },
                UpdateExpression="SET metrics = :metrics, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ':metrics': decimal_metrics,
                    ':updated_at': Decimal(str(datetime.utcnow().timestamp()))
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating model metrics: {str(e)}")
            return False 