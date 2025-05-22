from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import boto3
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from app.services.models.model_registry import ModelRegistry, ModelVersion
from app.services.evaluation.model_evaluator import ModelEvaluator, EvaluationConfig
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

class BenchmarkConfig(BaseModel):
    """Configuration for model benchmarking."""
    
    model_ids: List[str] = Field(..., description="List of model IDs to compare")
    versions: List[str] = Field(..., description="List of versions to compare")
    evaluation_data_path: str = Field(..., description="Path to evaluation data")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    batch_size: int = Field(default=32, description="Batch size for evaluation")
    comparison_name: str = Field(..., description="Name for this benchmark comparison")
    
    @validator('model_ids', 'versions')
    def validate_lists(cls, v):
        """Validate list lengths match."""
        if len(v) < 2:
            raise ValueError("Must provide at least 2 models/versions to compare")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate evaluation metrics."""
        valid_metrics = ["accuracy", "f1", "precision", "recall", "rouge", "bleu", "perplexity"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        return v

class BenchmarkResult(BaseModel):
    """Results from model benchmarking."""
    
    comparison_name: str
    timestamp: float
    models: List[Dict[str, Any]]
    metrics: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]

class ModelBenchmark:
    """Service for comparing and benchmarking models."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.model_registry = ModelRegistry(table_prefix=table_prefix)
        self.model_evaluator = ModelEvaluator(table_prefix=table_prefix)
        self.s3 = boto3.client('s3')
    
    def run_benchmark(self, config: BenchmarkConfig) -> str:
        """
        Run a benchmark comparison between multiple model versions.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            str: Benchmark job ID
        """
        try:
            # Create benchmark job
            job_id = f"benchmark-{config.comparison_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Evaluate each model version
            evaluation_jobs = []
            for model_id, version in zip(config.model_ids, config.versions):
                eval_config = EvaluationConfig(
                    model_id=model_id,
                    version=version,
                    evaluation_data_path=config.evaluation_data_path,
                    metrics=config.metrics,
                    batch_size=config.batch_size
                )
                
                eval_result = self.model_evaluator.evaluate_model(eval_config)
                evaluation_jobs.append({
                    'model_id': model_id,
                    'version': version,
                    'job_id': eval_result['job_id']
                })
            
            # Store benchmark job metadata
            self.model_registry.add_artifact(
                model_id=config.model_ids[0],  # Use first model as parent
                version=config.versions[0],    # Use first version as parent
                artifact_type="benchmark_job",
                s3_path=job_id,
                metadata={
                    'comparison_name': config.comparison_name,
                    'models': config.model_ids,
                    'versions': config.versions,
                    'evaluation_jobs': evaluation_jobs,
                    'status': 'running',
                    'started_at': datetime.utcnow().timestamp()
                }
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting benchmark job: {str(e)}")
            raise
    
    def get_benchmark_results(self, job_id: str) -> BenchmarkResult:
        """
        Get results from a benchmark comparison.
        
        Args:
            job_id: Benchmark job ID
            
        Returns:
            BenchmarkResult containing comparison results
        """
        try:
            # Get benchmark job metadata
            artifacts = self.model_registry.get_version_artifacts(
                model_id=job_id.split('-')[1],  # Extract model ID from job ID
                version=job_id.split('-')[2]    # Extract version from job ID
            )
            benchmark_job = next((a for a in artifacts if a.artifact_type == "benchmark_job" and a.s3_path == job_id), None)
            
            if not benchmark_job:
                raise ValueError(f"No benchmark job found with ID {job_id}")
            
            # Get evaluation results for each model
            models = []
            metrics = {}
            all_completed = True
            
            for eval_job in benchmark_job.metadata['evaluation_jobs']:
                eval_results = self.model_evaluator.get_evaluation_results(
                    eval_job['model_id'],
                    eval_job['version']
                )
                
                model_info = {
                    'model_id': eval_job['model_id'],
                    'version': eval_job['version'],
                    'status': eval_results['status']
                }
                
                if eval_results['status'] == 'Completed':
                    model_info['metrics'] = eval_results['results']['metrics']
                    metrics[f"{eval_job['model_id']}-{eval_job['version']}"] = eval_results['results']['metrics']
                else:
                    all_completed = False
                
                models.append(model_info)
            
            # Generate comparison summary
            summary = {}
            if all_completed:
                summary = self._generate_comparison_summary(metrics)
                
                # Update benchmark job status
                self.model_registry.add_artifact(
                    model_id=benchmark_job.model_id,
                    version=benchmark_job.version,
                    artifact_type="benchmark_results",
                    s3_path=job_id,
                    metadata={
                        'comparison_name': benchmark_job.metadata['comparison_name'],
                        'models': models,
                        'metrics': metrics,
                        'summary': summary,
                        'status': 'completed',
                        'completed_at': datetime.utcnow().timestamp()
                    }
                )
            
            return BenchmarkResult(
                comparison_name=benchmark_job.metadata['comparison_name'],
                timestamp=datetime.utcnow().timestamp(),
                models=models,
                metrics=metrics,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error getting benchmark results: {str(e)}")
            raise
    
    def _generate_comparison_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate a summary of model comparisons.
        
        Args:
            metrics: Dictionary of metrics for each model version
            
        Returns:
            Dict containing comparison summary
        """
        summary = {
            'best_models': {},
            'relative_performance': {},
            'statistical_significance': {}
        }
        
        # Find best model for each metric
        for metric in next(iter(metrics.values())).keys():
            best_model = max(metrics.items(), key=lambda x: x[1].get(metric, float('-inf')))
            summary['best_models'][metric] = {
                'model': best_model[0],
                'score': best_model[1][metric]
            }
        
        # Calculate relative performance
        for model, model_metrics in metrics.items():
            summary['relative_performance'][model] = {}
            for metric, score in model_metrics.items():
                best_score = summary['best_models'][metric]['score']
                relative_score = (score / best_score) * 100 if best_score != 0 else 0
                summary['relative_performance'][model][metric] = relative_score
        
        # Calculate statistical significance (simplified)
        for metric in next(iter(metrics.values())).keys():
            scores = [m[metric] for m in metrics.values()]
            mean = sum(scores) / len(scores)
            std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
            
            summary['statistical_significance'][metric] = {
                'mean': mean,
                'std': std,
                'cv': (std / mean) * 100 if mean != 0 else 0  # Coefficient of variation
            }
        
        return summary 