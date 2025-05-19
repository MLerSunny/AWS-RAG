"""
AWS Bedrock fine-tuning implementation.
"""
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, Sequence, cast, Callable, Type
import json
import time
import boto3
import os
import uuid
from enum import Enum
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError
from ...utils.logger import setup_logger
from ...utils.validation import validate_not_empty, validate_dict, validate_type, validate_range
from ...config import settings

logger = setup_logger(__name__)

class FineTuneError(Exception):
    """Base exception for fine-tuning errors."""
    def __init__(self, message: str, error_code: str = "FINETUNE_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class ValidationError(FineTuneError):
    """Raised when input validation fails."""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class JobError(FineTuneError):
    """Raised when job operation fails."""
    def __init__(self, message: str):
        super().__init__(message, "JOB_ERROR")

class ModelError(FineTuneError):
    """Raised when model operation fails."""
    def __init__(self, message: str):
        super().__init__(message, "MODEL_ERROR")

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
    
    # PEFT/LoRA configuration
    enable_peft: bool = False
    peft_method: str = "lora"  # Options: "lora", "prefix_tuning", "p_tuning"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    
    @validator('job_name')
    def validate_job_name(cls, v):
        """Validate job name."""
        if not v:
            raise ValidationError("Job name cannot be empty")
        if len(v) > 63:
            raise ValidationError("Job name must be 63 characters or less")
        if not v[0].isalnum():
            raise ValidationError("Job name must start with an alphanumeric character")
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValidationError("Job name can only contain alphanumeric characters, hyphens, and underscores")
        return v
    
    @validator('base_model_id')
    def validate_base_model_id(cls, v):
        """Validate base model ID."""
        if not v:
            raise ValidationError("Base model ID cannot be empty")
        return v
    
    @validator('training_data_path', 'output_data_path')
    def validate_s3_path(cls, v):
        """Validate S3 path."""
        if not v:
            raise ValidationError("S3 path cannot be empty")
        if not v.startswith('s3://'):
            raise ValidationError("S3 path must start with 's3://'")
        return v
    
    @validator('validation_data_path')
    def validate_validation_path(cls, v):
        """Validate validation data path."""
        if v and not v.startswith('s3://'):
            raise ValidationError("Validation data path must start with 's3://'")
        return v
    
    @validator('lora_rank')
    def validate_lora_rank(cls, v, values):
        """Validate LoRA rank."""
        if values.get('enable_peft') and values.get('peft_method') == 'lora':
            if v < 1:
                raise ValidationError("LoRA rank must be at least 1")
            if v > 64:
                raise ValidationError("LoRA rank must be at most 64")
        return v
    
    @validator('lora_alpha')
    def validate_lora_alpha(cls, v, values):
        """Validate LoRA alpha."""
        if values.get('enable_peft') and values.get('peft_method') == 'lora':
            if v <= 0:
                raise ValidationError("LoRA alpha must be positive")
        return v
    
    @validator('lora_dropout')
    def validate_lora_dropout(cls, v, values):
        """Validate LoRA dropout."""
        if values.get('enable_peft') and values.get('peft_method') == 'lora':
            if v < 0 or v > 1:
                raise ValidationError("LoRA dropout must be between 0 and 1")
        return v

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
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize the Bedrock fine-tuner.
        
        Args:
            region (Optional[str]): AWS region
            
        Raises:
            FineTuneError: If initialization fails
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
            raise FineTuneError(f"Failed to initialize Bedrock fine-tuner: {str(e)}")
    
    def _retry_operation(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation (Callable[..., Any]): Operation to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Any: Result of the operation
            
        Raises:
            FineTuneError: If operation fails after retries
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                last_error = e
                if attempt == self.MAX_RETRIES - 1:
                    break
                delay = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Operation failed, retrying in {delay}s: {str(e)}")
                time.sleep(delay)
            except Exception as e:
                raise FineTuneError(f"Operation failed: {str(e)}")
        
        raise FineTuneError(f"Operation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
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
            
        Raises:
            FineTuneError: If operation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
        try:
            response = self._retry_operation(
                self.bedrock.list_foundation_models
            )
            
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
            raise FineTuneError(f"Failed to list supported models: {str(e)}")
    
    def upload_training_file(self, file_path: str, bucket: str, key: str) -> bool:
        """
        Upload a training file to S3.
        
        Args:
            file_path (str): Local path to the file
            bucket (str): S3 bucket name
            key (str): S3 key (path)
            
        Returns:
            bool: True if successful
            
        Raises:
            FineTuneError: If operation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
        try:
            self._retry_operation(
                self.s3.upload_file,
                file_path,
                bucket,
                key
            )
            logger.info(f"Uploaded training file to s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading training file: {str(e)}")
            raise FineTuneError(f"Failed to upload training file: {str(e)}")
    
    def create_fine_tune_job(self, config: FineTuneConfig) -> FineTuneJob:
        """
        Create a fine-tuning job.
        
        Args:
            config (FineTuneConfig): Fine-tuning configuration
            
        Returns:
            FineTuneJob: Fine-tuning job
            
        Raises:
            ValidationError: If configuration is invalid
            JobError: If job creation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
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
            
            # Add PEFT/LoRA parameters if enabled
            if config.enable_peft:
                hyperparameters.update({
                    "enable_peft": "true",
                    "peft_method": config.peft_method,
                    "lora_rank": str(config.lora_rank),
                    "lora_alpha": str(config.lora_alpha),
                    "lora_dropout": str(config.lora_dropout),
                    "target_modules": ",".join(config.target_modules)
                })
                logger.info(f"Enabling {config.peft_method} with rank {config.lora_rank} for fine-tuning")
            
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
                
            if config.description:
                job_params["description"] = config.description
            
            # Create job
            response = self._retry_operation(
                self.bedrock.create_model_customization_job,
                **job_params
            )
            
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
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            raise JobError(f"Failed to create fine-tuning job: {str(e)}")
    
    def get_job_status(self, job_id: str) -> FineTuneJob:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            FineTuneJob: Updated job information
            
        Raises:
            JobError: If operation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
        try:
            response = self._retry_operation(
                self.bedrock.get_model_customization_job,
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
            raise JobError(f"Failed to get job status: {str(e)}")
    
    def stop_fine_tune_job(self, job_id: str) -> bool:
        """
        Stop a fine-tuning job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if successful
            
        Raises:
            JobError: If operation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
        try:
            self._retry_operation(
                self.bedrock.stop_model_customization_job,
                jobIdentifier=job_id
            )
            
            logger.info(f"Stopped fine-tuning job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping fine-tuning job: {str(e)}")
            raise JobError(f"Failed to stop fine-tuning job: {str(e)}")
    
    def delete_custom_model(self, model_id: str) -> bool:
        """
        Delete a custom fine-tuned model.
        
        Args:
            model_id (str): Custom model ID
            
        Returns:
            bool: True if successful
            
        Raises:
            ModelError: If operation fails
        """
        if not self.is_initialized():
            raise FineTuneError("Bedrock fine-tuner not initialized")
            
        try:
            self._retry_operation(
                self.bedrock.delete_custom_model,
                modelIdentifier=model_id
            )
            
            logger.info(f"Deleted custom model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting custom model: {str(e)}")
            raise ModelError(f"Failed to delete custom model: {str(e)}")
    
    def generate_jsonl_format(self, examples: List[Dict[str, str]], output_path: str) -> bool:
        """
        Generate JSONL format for fine-tuning.
        
        Args:
            examples (List[Dict]): List of examples with 'input' and 'output' keys
            output_path (str): Path to save the JSONL file
            
        Returns:
            bool: True if successful
            
        Raises:
            ValidationError: If input is invalid
            FineTuneError: If operation fails
        """
        try:
            # Validate examples
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    raise ValidationError(f"Example {i} must be a dictionary")
                if 'input' not in example:
                    raise ValidationError(f"Example {i} missing 'input' key")
                if 'output' not in example:
                    raise ValidationError(f"Example {i} missing 'output' key")
                if not isinstance(example['input'], str):
                    raise ValidationError(f"Example {i} 'input' must be a string")
                if not isinstance(example['output'], str):
                    raise ValidationError(f"Example {i} 'output' must be a string")
            
            with open(output_path, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
                    
            logger.info(f"Generated JSONL file with {len(examples)} examples at {output_path}")
            return True
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error generating JSONL file: {str(e)}")
            raise FineTuneError(f"Failed to generate JSONL file: {str(e)}")
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parse an S3 path into bucket and key.
        
        Args:
            s3_path (str): S3 path in the format 's3://bucket/key'
            
        Returns:
            tuple: (bucket, key)
            
        Raises:
            ValidationError: If path is invalid
        """
        if not s3_path.startswith('s3://'):
            raise ValidationError("S3 path must start with 's3://'")
            
        path = s3_path[5:]
        parts = path.split('/')
        
        if len(parts) < 2:
            raise ValidationError("Invalid S3 path format")
            
        bucket = parts[0]
        key = '/'.join(parts[1:])
        
        return bucket, key

    @staticmethod
    def create_example_config(method: str = "standard", model_id: str = "anthropic.claude-v2") -> FineTuneConfig:
        """
        Create an example fine-tuning configuration.
        
        Args:
            method (str): Fine-tuning method - 'standard', 'lora', or 'combined'
            model_id (str): Base model ID
            
        Returns:
            FineTuneConfig: Example configuration
            
        Raises:
            ValidationError: If method is invalid
        """
        if method not in ["standard", "lora", "combined"]:
            raise ValidationError("Method must be 'standard', 'lora', or 'combined'")
            
        # Base configuration
        config = FineTuneConfig(
            job_name=f"{method}-finetune-{int(time.time())}",
            base_model_id=model_id,
            training_data_path="s3://your-bucket/training.jsonl",
            output_data_path="s3://your-bucket/output/",
            description=f"Example {method} fine-tuning configuration"
        )
        
        # Configure based on method
        if method == "standard":
            # Standard fine-tuning with common hyperparameters
            config.hyperparameters = {
                "epochCount": "3",
                "batchSize": "1",
                "learningRate": "0.0001",
                "weightDecay": "0.01",
                "warmupSteps": "100"
            }
            config.enable_peft = False
            
        elif method == "lora":
            # LoRA-only fine-tuning with minimal full model training
            config.hyperparameters = {
                "epochCount": "3", 
                "batchSize": "4",  # Can use larger batches with LoRA
                "learningRate": "0.0003",
                "weightDecay": "0.01"
            }
            config.enable_peft = True
            config.peft_method = "lora"
            config.lora_rank = 8
            config.lora_alpha = 16.0
            config.lora_dropout = 0.05
            config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
        elif method == "combined":
            # Combined approach with tailored hyperparameters
            config.hyperparameters = {
                "epochCount": "4",
                "batchSize": "2",
                "learningRate": "0.0001",
                "weightDecay": "0.01",
                "warmupSteps": "200",
                "scheduler": "cosine"  # Custom scheduler
            }
            config.enable_peft = True
            config.peft_method = "lora"
            config.lora_rank = 16  # Higher rank for better capacity
            config.lora_alpha = 32.0
            config.lora_dropout = 0.1
            config.target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        
        return config 