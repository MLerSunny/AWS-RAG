"""
AWS Bedrock fine-tuning implementation.
"""
from typing import Dict, List, Optional, Any, Union
import json
import time
import boto3
import os
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from ...utils.logger import setup_logger
from ...config import settings

logger = setup_logger(__name__)

class FineTuneStatus(str, Enum):
    """Status of a fine-tuning job."""
    PREPARING = "preparing"    # Preparing data
    PENDING = "pending"        # Waiting for AWS to start
    RUNNING = "running"        # In progress
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Failed
    STOPPED = "stopped"        # Manually stopped

class TrainingDataFormat(str, Enum):
    """Supported training data formats."""
    JSONL = "jsonl"
    CSV = "csv"

class FineTuneConfig(BaseModel):
    """Configuration for fine-tuning a model."""
    job_name: str
    base_model_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_data_path: str
    validation_data_path: Optional[str] = None
    output_data_path: str
    data_format: TrainingDataFormat = TrainingDataFormat.JSONL
    custom_model_name: Optional[str] = None
    description: Optional[str] = None

class FineTuneJob(BaseModel):
    """Representation of a fine-tuning job."""
    id: str
    job_name: str
    base_model_id: str
    custom_model_id: Optional[str] = None
    status: FineTuneStatus = FineTuneStatus.PREPARING
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    description: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    job_arn: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "job_name": self.job_name,
            "base_model_id": self.base_model_id,
            "custom_model_id": self.custom_model_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "description": self.description,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "job_arn": self.job_arn
        }

class BedrockFineTuner:
    """Service for fine-tuning AWS Bedrock models."""
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize the Bedrock fine-tuner.
        
        Args:
            region (Optional[str]): AWS region
        """
        self.region = region or settings.AWS_REGION
        
        try:
            # Initialize AWS clients
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            
            self.bedrock = session.client(service_name='bedrock')
            self.bedrock_runtime = session.client(service_name='bedrock-runtime')
            self.s3 = session.client(service_name='s3')
            
            logger.info(f"Initialized Bedrock fine-tuner in region {self.region}")
        except Exception as e:
            logger.error(f"Error initializing Bedrock fine-tuner: {str(e)}")
            self.bedrock = None
            self.bedrock_runtime = None
            self.s3 = None
    
    def is_initialized(self) -> bool:
        """
        Check if the service is properly initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.bedrock is not None and self.bedrock_runtime is not None and self.s3 is not None
    
    def get_supported_models(self) -> List[Dict[str, Any]]:
        """
        Get list of Bedrock models that support fine-tuning.
        
        Returns:
            List[Dict]: List of supported models
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return []
            
        try:
            response = self.bedrock.list_foundation_models()
            
            # Filter to only models supporting fine-tuning
            supported_models = []
            for model in response.get("modelSummaries", []):
                if model.get("customizationsSupported", {}).get("finetuning", False):
                    supported_models.append({
                        "model_id": model.get("modelId"),
                        "name": model.get("modelName"),
                        "provider": model.get("providerName"),
                        "input_modalities": model.get("inputModalities", []),
                        "output_modalities": model.get("outputModalities", [])
                    })
            
            logger.info(f"Found {len(supported_models)} models supporting fine-tuning")
            return supported_models
        except Exception as e:
            logger.error(f"Error listing supported models: {str(e)}")
            return []
    
    def upload_training_file(self, file_path: str, bucket: str, key: str) -> bool:
        """
        Upload a training file to S3.
        
        Args:
            file_path (str): Local path to the file
            bucket (str): S3 bucket name
            key (str): S3 key (path)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return False
            
        try:
            self.s3.upload_file(file_path, bucket, key)
            logger.info(f"Uploaded training file to s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading training file: {str(e)}")
            return False
    
    def create_fine_tune_job(self, config: FineTuneConfig) -> Optional[FineTuneJob]:
        """
        Create a fine-tuning job.
        
        Args:
            config (FineTuneConfig): Fine-tuning configuration
            
        Returns:
            Optional[FineTuneJob]: Fine-tuning job or None if failed
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return None
            
        try:
            # Parse S3 paths
            training_bucket, training_key = self._parse_s3_path(config.training_data_path)
            output_bucket, output_key = self._parse_s3_path(config.output_data_path)
            
            # Build job name if not specified
            job_name = config.job_name
            if not job_name:
                job_name = f"finetune-{config.base_model_id.split('/')[-1]}-{int(time.time())}"
            
            # Build hyperparameters
            hyperparameters = config.hyperparameters.copy()
            # Add defaults if not provided
            if "epochCount" not in hyperparameters:
                hyperparameters["epochCount"] = "3"
            if "batchSize" not in hyperparameters:
                hyperparameters["batchSize"] = "1"
            if "learningRate" not in hyperparameters:
                hyperparameters["learningRate"] = "0.0001"
            
            # Convert hyperparameters to strings
            for key, value in hyperparameters.items():
                hyperparameters[key] = str(value)
                
            # Prepare job request
            job_params = {
                "jobName": job_name,
                "baseModelIdentifier": config.base_model_id,
                "trainingDataConfig": {
                    "s3Uri": f"s3://{training_bucket}/{training_key}"
                },
                "outputDataConfig": {
                    "s3Uri": f"s3://{output_bucket}/{output_key}"
                },
                "hyperParameters": hyperparameters
            }
            
            # Add optional parameters
            if config.validation_data_path:
                val_bucket, val_key = self._parse_s3_path(config.validation_data_path)
                job_params["validationDataConfig"] = {
                    "s3Uri": f"s3://{val_bucket}/{val_key}"
                }
                
            if config.custom_model_name:
                job_params["customModelName"] = config.custom_model_name
                
            # Create job
            response = self.bedrock.create_model_customization_job(**job_params)
            
            # Parse response
            job_id = response.get("jobArn").split("/")[-1]
            
            # Create job object
            job = FineTuneJob(
                id=job_id,
                job_name=job_name,
                base_model_id=config.base_model_id,
                status=FineTuneStatus.PENDING,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                description=config.description,
                job_arn=response.get("jobArn")
            )
            
            logger.info(f"Created fine-tuning job {job_id} for model {config.base_model_id}")
            return job
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[FineTuneJob]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            Optional[FineTuneJob]: Updated job information or None if failed
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return None
            
        try:
            response = self.bedrock.get_model_customization_job(
                jobIdentifier=job_id
            )
            
            # Parse status
            aws_status = response.get("status")
            
            if aws_status == "InProgress":
                status = FineTuneStatus.RUNNING
            elif aws_status == "Completed":
                status = FineTuneStatus.COMPLETED
            elif aws_status == "Failed":
                status = FineTuneStatus.FAILED
            elif aws_status == "Stopping":
                status = FineTuneStatus.STOPPED
            elif aws_status == "Stopped":
                status = FineTuneStatus.STOPPED
            else:
                status = FineTuneStatus.PENDING
            
            # Create job object
            job = FineTuneJob(
                id=job_id,
                job_name=response.get("jobName"),
                base_model_id=response.get("baseModelIdentifier"),
                custom_model_id=response.get("outputModelArn"),
                status=status,
                created_at=response.get("creationTime"),
                started_at=response.get("trainingStartTime"),
                completed_at=response.get("endTime"),
                metrics={},
                error_message=response.get("failureMessage"),
                job_arn=response.get("jobArn")
            )
            
            logger.info(f"Job {job_id} status: {status}")
            return job
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None
    
    def stop_fine_tune_job(self, job_id: str) -> bool:
        """
        Stop a fine-tuning job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return False
            
        try:
            self.bedrock.stop_model_customization_job(
                jobIdentifier=job_id
            )
            
            logger.info(f"Stopped fine-tuning job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping fine-tuning job: {str(e)}")
            return False
    
    def delete_custom_model(self, model_id: str) -> bool:
        """
        Delete a custom fine-tuned model.
        
        Args:
            model_id (str): Custom model ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized():
            logger.error("Bedrock fine-tuner not initialized")
            return False
            
        try:
            self.bedrock.delete_custom_model(
                modelIdentifier=model_id
            )
            
            logger.info(f"Deleted custom model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting custom model: {str(e)}")
            return False
    
    def generate_jsonl_format(self, examples: List[Dict[str, str]], output_path: str) -> bool:
        """
        Generate JSONL format for fine-tuning.
        
        Args:
            examples (List[Dict]): List of examples with 'input' and 'output' keys
            output_path (str): Path to save the JSONL file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
                    
            logger.info(f"Generated JSONL file with {len(examples)} examples at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating JSONL file: {str(e)}")
            return False
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parse an S3 path into bucket and key.
        
        Args:
            s3_path (str): S3 path in the format 's3://bucket/key'
            
        Returns:
            tuple: (bucket, key)
        """
        if s3_path.startswith('s3://'):
            path = s3_path[5:]
        else:
            path = s3_path
            
        parts = path.split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        
        return bucket, key 