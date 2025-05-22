from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import boto3
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from app.services.models.model_registry import ModelRegistry, ModelVersion
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    model_id: str = Field(..., description="ID of the model to evaluate")
    version: str = Field(..., description="Version identifier")
    evaluation_data_path: str = Field(..., description="Path to evaluation data")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    batch_size: int = Field(default=32, description="Batch size for evaluation")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate evaluation metrics."""
        valid_metrics = ["accuracy", "f1", "precision", "recall", "rouge", "bleu", "perplexity"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        return v

class ModelEvaluator:
    """Service for evaluating model performance."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.model_registry = ModelRegistry(table_prefix=table_prefix)
        self.bedrock = boto3.client('bedrock')
        self.s3 = boto3.client('s3')
    
    def evaluate_model(self, config: EvaluationConfig) -> Dict[str, Any]:
        """
        Evaluate a model version using specified metrics.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Get model version
            version = self.model_registry.get_version(config.model_id, config.version)
            if not version:
                raise ValueError(f"Model version {config.version} not found")
            
            # Get model artifacts
            artifacts = self.model_registry.get_version_artifacts(config.model_id, config.version)
            model_artifact = next((a for a in artifacts if a.artifact_type == "model"), None)
            if not model_artifact:
                raise ValueError(f"No model artifact found for version {config.version}")
            
            # Upload evaluation data to S3
            eval_s3_path = self._upload_to_s3(
                config.evaluation_data_path,
                f"evaluation/{config.model_id}/{config.version}/eval.jsonl"
            )
            
            # Create evaluation job
            job_name = f"eval-{config.model_id}-{config.version}"
            job_config = {
                "jobName": job_name,
                "modelId": config.model_id,
                "evaluationDataConfig": {
                    "s3Uri": eval_s3_path
                },
                "evaluationMetrics": config.metrics,
                "batchSize": config.batch_size
            }
            
            response = self.bedrock.create_model_evaluation_job(**job_config)
            job_id = response['jobId']
            
            # Add evaluation job as artifact
            self.model_registry.add_artifact(
                model_id=config.model_id,
                version=config.version,
                artifact_type="evaluation_job",
                s3_path=job_id,
                metadata={
                    "job_name": job_name,
                    "status": "evaluating",
                    "metrics": config.metrics,
                    "started_at": datetime.utcnow().timestamp()
                }
            )
            
            return {
                'job_id': job_id,
                'status': 'evaluating',
                'metrics': config.metrics
            }
            
        except Exception as e:
            logger.error(f"Error starting evaluation job: {str(e)}")
            raise
    
    def get_evaluation_results(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        Get evaluation results for a model version.
        
        Args:
            model_id: ID of the model
            version: Version identifier
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Get evaluation job artifact
            artifacts = self.model_registry.get_version_artifacts(model_id, version)
            eval_job = next((a for a in artifacts if a.artifact_type == "evaluation_job"), None)
            
            if not eval_job:
                raise ValueError(f"No evaluation job found for model {model_id} version {version}")
            
            # Get job status from Bedrock
            response = self.bedrock.get_model_evaluation_job(
                jobId=eval_job.s3_path
            )
            
            status = response['status']
            results = {}
            
            if status == 'Completed':
                results = {
                    'metrics': response.get('evaluationMetrics', {}),
                    'summary': response.get('evaluationSummary', {}),
                    'completed_at': datetime.utcnow().timestamp()
                }
                
                # Update model version metrics
                self.model_registry.update_version_metrics(model_id, version, results['metrics'])
                
                # Add evaluation results as artifact
                self.model_registry.add_artifact(
                    model_id=model_id,
                    version=version,
                    artifact_type="evaluation_results",
                    s3_path=eval_job.s3_path,
                    metadata=results
                )
            
            return {
                'status': status,
                'results': results,
                'job_id': eval_job.s3_path
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation results: {str(e)}")
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