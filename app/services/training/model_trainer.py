from typing import Dict, List, Optional, Any, Union, TypedDict, cast
from datetime import datetime
import json
import os
import boto3
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from app.services.models.model_registry import ModelRegistry, ModelVersion, ModelArtifact
from app.services.training.training_strategies import (
    TrainingStrategy,
    StrategyConfig,
    get_strategy_config,
    get_hyperparameters
)
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

class StrategyConfigDict(TypedDict, total=False):
    """Type definition for strategy configuration dictionary."""
    learning_rate: float
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    unfreeze_layers: Optional[int]
    target_modules: List[str]
    r: int
    alpha: int
    dropout: float
    use_rslora: bool
    use_dora: bool
    bits: int
    double_quant: bool
    quant_type: str
    memory_size: int
    replay_strategy: str
    importance_sampling: bool

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    model_id: str = Field(..., description="ID of the model to train")
    version: str = Field(..., description="Version identifier")
    training_data_path: str = Field(..., description="Path to training data")
    validation_data_path: Optional[str] = Field(None, description="Path to validation data")
    strategy: TrainingStrategy = Field(default=TrainingStrategy.LORA, description="Training strategy to use")
    strategy_config: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific configuration")
    output_path: str = Field(..., description="Path to save model artifacts")
    
    @validator('version')
    def validate_version(cls, v):
        """Validate version format."""
        if not v.startswith('v'):
            raise ValueError("Version must start with 'v'")
        try:
            parts = v[1:].split('.')
            if len(parts) != 3:
                raise ValueError
            [int(x) for x in parts]
        except (ValueError, IndexError):
            raise ValueError("Version must be in format vX.Y.Z where X, Y, Z are numbers")
        return v

class ModelTrainer:
    """Service for training and fine-tuning models."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.model_registry = ModelRegistry(table_prefix=table_prefix)
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client('bedrock')
        
    def start_training(self, config: TrainingConfig) -> str:
        """
        Start a model training job.
        
        Args:
            config: Training configuration
            
        Returns:
            str: Training job ID
        """
        try:
            # Get strategy configuration
            strategy_config = get_strategy_config(
                config.strategy,
                **cast(Dict[str, Any], config.strategy_config)
            )
            hyperparameters = get_hyperparameters(strategy_config)
            
            # Register model version
            version = self.model_registry.register_version(
                model_id=config.model_id,
                version=config.version,
                hyperparameters=hyperparameters
            )
            
            # Upload training data to S3
            training_s3_path = self._upload_to_s3(
                config.training_data_path,
                f"training/{config.model_id}/{config.version}/train.jsonl"
            )
            
            validation_s3_path = None
            if config.validation_data_path:
                validation_s3_path = self._upload_to_s3(
                    config.validation_data_path,
                    f"training/{config.model_id}/{config.version}/validation.jsonl"
                )
            
            # Create Bedrock fine-tuning job
            job_name = f"{config.model_id}-{config.version}"
            job_config = {
                "jobName": job_name,
                "modelId": config.model_id,
                "trainingDataConfig": {
                    "s3Uri": training_s3_path
                },
                "outputDataConfig": {
                    "s3Uri": f"s3://{settings.S3_BUCKET}/{config.output_path}"
                },
                "hyperParameters": hyperparameters,
                "trainingStrategy": config.strategy.value
            }
            
            if validation_s3_path:
                job_config["validationDataConfig"] = {
                    "s3Uri": validation_s3_path
                }
            
            response = self.bedrock.create_model_customization_job(**job_config)
            job_id = response['jobId']
            
            # Add training job as artifact
            self.model_registry.add_artifact(
                model_id=config.model_id,
                version=config.version,
                artifact_type="training_job",
                s3_path=job_id,
                metadata={
                    "job_name": job_name,
                    "status": "training",
                    "strategy": config.strategy.value,
                    "started_at": datetime.utcnow().timestamp()
                }
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting training job: {str(e)}")
            raise
    
    def check_training_status(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        Check the status of a training job.
        
        Args:
            model_id: ID of the model
            version: Version identifier
            
        Returns:
            Dict containing training status and metrics
        """
        try:
            # Get training job artifact
            artifacts = self.model_registry.get_version_artifacts(model_id, version)
            training_job = next((a for a in artifacts if a.artifact_type == "training_job"), None)
            
            if not training_job:
                raise ValueError(f"No training job found for model {model_id} version {version}")
            
            # Get job status from Bedrock
            response = self.bedrock.get_model_customization_job(
                jobId=training_job.s3_path
            )
            
            status = response['status']
            metrics = {}
            
            if status == 'Completed':
                metrics = {
                    'training_loss': float(response.get('trainingMetrics', {}).get('trainingLoss', 0)),
                    'validation_loss': float(response.get('validationMetrics', {}).get('validationLoss', 0))
                }
                
                # Update model version status and metrics
                self.model_registry.update_version_status(model_id, version, 'active')
                self.model_registry.update_version_metrics(model_id, version, metrics)
                
                # Add model artifacts
                model_arn = response.get('outputModelArn')
                if model_arn:
                    self.model_registry.add_artifact(
                        model_id=model_id,
                        version=version,
                        artifact_type="model",
                        s3_path=model_arn,
                        metadata={
                            "status": "active",
                            "completed_at": datetime.utcnow().timestamp()
                        }
                    )
            
            return {
                'status': status,
                'metrics': metrics,
                'job_id': training_job.s3_path
            }
            
        except Exception as e:
            logger.error(f"Error checking training status: {str(e)}")
            raise
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            
        Returns:
            str: S3 URI
        """
        try:
            self.s3.upload_file(local_path, settings.S3_BUCKET, s3_key)
            return f"s3://{settings.S3_BUCKET}/{s3_key}"
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise 