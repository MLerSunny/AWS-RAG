"""
Enhanced AWS Bedrock fine-tuning with early divergence detection.

This module extends the base Bedrock fine-tuner with monitoring capabilities
to detect training issues early.
"""
import json
import time
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, TypeVar, Generic, Sequence, cast, Callable, Type
import concurrent.futures
import boto3
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

from .bedrock_finetune import BedrockFineTuner, FineTuneJob, FineTuneConfig, FineTuneStatus, FineTuneError
from .training_monitor import TrainingMonitor, MonitoringConfig, MetricType, MetricStatus
from ...utils.logger import setup_logger
from ...utils.validation import validate_not_empty, validate_dict, validate_type, validate_range

logger = setup_logger(__name__)

class MonitoringError(Exception):
    """Base exception for monitoring errors."""
    def __init__(self, message: str, error_code: str = "MONITORING_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class ValidationError(MonitoringError):
    """Raised when input validation fails."""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

class JobError(MonitoringError):
    """Raised when job operation fails."""
    def __init__(self, message: str):
        super().__init__(message, "JOB_ERROR")

class BedrockMonitoringJob:
    """Job for monitoring a Bedrock fine-tuning process."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self, 
                 job_id: str, 
                 monitor_config: MonitoringConfig,
                 poll_interval_seconds: int = 60,
                 metrics_log_path: Optional[str] = None):
        """
        Initialize a monitoring job.
        
        Args:
            job_id: Bedrock job ID to monitor
            monitor_config: Configuration for monitoring
            poll_interval_seconds: How often to poll for status
            metrics_log_path: Path to save metrics logs
            
        Raises:
            ValidationError: If input parameters are invalid
        """
        validate_not_empty(job_id, "job_id")
        validate_type(monitor_config, MonitoringConfig, "monitor_config")
        validate_range(poll_interval_seconds, "poll_interval_seconds", 1, 3600)
        
        self.job_id = job_id
        self.monitor = TrainingMonitor(monitor_config)
        self.poll_interval = poll_interval_seconds
        self.metrics_log_path = metrics_log_path
        self.should_continue = True
        self.thread = None
        self.last_metrics_save_time = 0
        self.save_interval_seconds = 300  # Save metrics every 5 minutes
        self.last_status = None
        self.issues_detected = []
        self.recovery_actions = []
    
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
            MonitoringError: If operation fails after retries
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
                raise MonitoringError(f"Operation failed: {str(e)}")
        
        raise MonitoringError(f"Operation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    def start(self, bedrock_finetuner: BedrockFineTuner) -> None:
        """
        Start the monitoring job in a separate thread.
        
        Args:
            bedrock_finetuner: BedrockFineTuner instance
            
        Raises:
            MonitoringError: If monitoring fails to start
        """
        if self.thread is not None and self.thread.is_alive():
            raise MonitoringError(f"Monitoring job for {self.job_id} already running")
            
        try:
            self.should_continue = True
            self.thread = threading.Thread(
                target=self._monitoring_loop,
                args=(bedrock_finetuner,),
                daemon=True
            )
            self.thread.start()
            logger.info(f"Started monitoring thread for job {self.job_id}")
        except Exception as e:
            logger.error(f"Error starting monitoring thread: {str(e)}")
            raise MonitoringError(f"Failed to start monitoring thread: {str(e)}")
    
    def stop(self) -> None:
        """
        Stop the monitoring job.
        
        Raises:
            MonitoringError: If monitoring fails to stop
        """
        try:
            self.should_continue = False
            if self.thread is not None and self.thread.is_alive():
                # Wait for the thread to finish with timeout
                self.thread.join(timeout=10)
                logger.info(f"Stopped monitoring thread for job {self.job_id}")
        except Exception as e:
            logger.error(f"Error stopping monitoring thread: {str(e)}")
            raise MonitoringError(f"Failed to stop monitoring thread: {str(e)}")
    
    def _monitoring_loop(self, bedrock_finetuner: BedrockFineTuner) -> None:
        """
        Main monitoring loop.
        
        Args:
            bedrock_finetuner: BedrockFineTuner instance
            
        Raises:
            MonitoringError: If monitoring fails
        """
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while self.should_continue:
            try:
                # Get job status
                job = self._retry_operation(
                    bedrock_finetuner.get_job_status,
                    self.job_id
                )
                
                # Track status changes
                if self.last_status != job.status:
                    logger.info(f"Job {self.job_id} status changed: {self.last_status} -> {job.status}")
                    self.last_status = job.status
                
                # Stop monitoring if job is complete or failed
                if job.status in [FineTuneStatus.COMPLETED, FineTuneStatus.FAILED, FineTuneStatus.STOPPED]:
                    logger.info(f"Job {self.job_id} is {job.status}. Stopping monitoring.")
                    self._save_metrics_log(job)
                    self.should_continue = False
                    break
                
                # Extract and track metrics
                self._process_job_metrics(job)
                
                # Check for early stopping conditions
                should_stop, reason = self.monitor.should_stop_training()
                if should_stop:
                    logger.warning(f"Recommended stopping job {self.job_id}: {reason}")
                    self.issues_detected.append({
                        "timestamp": time.time(),
                        "issue": "training_divergence",
                        "reason": reason,
                        "action": "stop_recommended"
                    })
                    
                    # Stop the job if auto-stop is enabled
                    if self.monitor.config.auto_stop_enabled:
                        logger.warning(f"Auto-stopping job {self.job_id} due to: {reason}")
                        self._retry_operation(
                            bedrock_finetuner.stop_fine_tune_job,
                            self.job_id
                        )
                        self.recovery_actions.append({
                            "timestamp": time.time(),
                            "action": "stop_job",
                            "reason": reason
                        })
                
                # Periodically save metrics
                current_time = time.time()
                if current_time - self.last_metrics_save_time > self.save_interval_seconds:
                    self._save_metrics_log(job)
                    self.last_metrics_save_time = current_time
                
                # Sleep before next poll
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for job {self.job_id}: {str(e)}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors. Stopping monitoring for job {self.job_id}")
                    self.should_continue = False
                time.sleep(self.poll_interval)
    
    def _process_job_metrics(self, job: FineTuneJob) -> None:
        """
        Process and track metrics from a fine-tuning job.
        
        Args:
            job: Current job state
            
        Raises:
            MonitoringError: If metric processing fails
        """
        try:
            # Extract CloudWatch metrics if available
            metrics = job.metrics
            
            # Track basic metrics if available
            if "trainingLoss" in metrics:
                self.monitor.track_metric("loss", float(metrics["trainingLoss"]), MetricType.LOSS)
            
            if "validationLoss" in metrics and "trainingLoss" in metrics:
                self.monitor.check_validation_metrics(
                    float(metrics["validationLoss"]), 
                    float(metrics["trainingLoss"])
                )
            
            # Calculate derivative metrics to detect issues
            if "trainingLoss" in metrics and "stepCount" in metrics:
                # Check loss curve health
                is_ok, message, status = self.monitor.check_loss_curve()
                if status in [MetricStatus.WARNING, MetricStatus.CRITICAL]:
                    logger.warning(f"Loss issue detected in job {self.job_id}: {message}")
                    self.issues_detected.append({
                        "timestamp": time.time(),
                        "issue": "loss_problem",
                        "details": message,
                        "status": status
                    })
            
            # Track resource metrics
            if "stepTimeSeconds" in metrics:
                self.monitor.track_metric(
                    "step_time", 
                    float(metrics["stepTimeSeconds"]), 
                    MetricType.RESOURCE
                )
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            raise MonitoringError(f"Failed to process metrics: {str(e)}")
    
    def _save_metrics_log(self, job: FineTuneJob) -> None:
        """
        Save current metrics to log file.
        
        Args:
            job: Current job state
            
        Raises:
            MonitoringError: If saving metrics fails
        """
        if not self.metrics_log_path:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metrics_log_path), exist_ok=True)
            
            # Prepare log data
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job_id": self.job_id,
                "status": job.status,
                "metrics": self.monitor.get_metrics_summary(),
                "job_metrics": job.metrics,
                "issues_detected": self.issues_detected,
                "recovery_actions": self.recovery_actions
            }
            
            # Append to log file
            with open(self.metrics_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving metrics log: {str(e)}")
            raise MonitoringError(f"Failed to save metrics log: {str(e)}")


class EnhancedBedrockFineTuner(BedrockFineTuner):
    """Enhanced Bedrock fine-tuner with monitoring capabilities."""
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize the enhanced fine-tuner.
        
        Args:
            region: AWS region
            
        Raises:
            FineTuneError: If initialization fails
        """
        # Initialize base class
        super().__init__(region)
        
        # Initialize monitoring components
        self.active_monitors: Dict[str, BedrockMonitoringJob] = {}
        self.default_monitor_config = MonitoringConfig(
            enabled=True,
            auto_stop_enabled=False,  # Default to not auto-stopping
            auto_lr_reduction=False   # AWS Bedrock doesn't support dynamic LR changes
        )
    
    def create_fine_tune_job(self, config: FineTuneConfig, 
                            enable_monitoring: bool = True,
                            monitor_config: Optional[MonitoringConfig] = None,
                            metrics_log_dir: Optional[str] = None) -> FineTuneJob:
        """
        Create a fine-tuning job with monitoring.
        
        Args:
            config: Fine-tuning configuration
            enable_monitoring: Whether to enable monitoring
            monitor_config: Custom monitoring configuration
            metrics_log_dir: Directory to save metrics logs
            
        Returns:
            FineTuneJob: Fine-tuning job
            
        Raises:
            ValidationError: If configuration is invalid
            JobError: If job creation fails
            MonitoringError: If monitoring setup fails
        """
        try:
            # Create the job using the base implementation
            job = super().create_fine_tune_job(config)
            
            # Set up monitoring if enabled
            if enable_monitoring:
                self._setup_job_monitoring(
                    job.id, 
                    monitor_config or self.default_monitor_config, 
                    metrics_log_dir
                )
                
            return job
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            raise JobError(f"Failed to create fine-tuning job: {str(e)}")
    
    def _setup_job_monitoring(self, job_id: str, 
                            monitor_config: MonitoringConfig,
                            metrics_log_dir: Optional[str] = None) -> None:
        """
        Set up monitoring for a job.
        
        Args:
            job_id: Job ID to monitor
            monitor_config: Monitoring configuration
            metrics_log_dir: Directory to save metrics logs
            
        Raises:
            ValidationError: If configuration is invalid
            MonitoringError: If monitoring setup fails
        """
        try:
            # Skip if already monitoring
            if job_id in self.active_monitors:
                logger.warning(f"Already monitoring job {job_id}")
                return
                
            # Set up metrics log path
            metrics_log_path = None
            if metrics_log_dir:
                os.makedirs(metrics_log_dir, exist_ok=True)
                metrics_log_path = os.path.join(
                    metrics_log_dir,
                    f"finetune_metrics_{job_id}_{int(time.time())}.jsonl"
                )
                
            # Create and start the monitoring job
            monitor_job = BedrockMonitoringJob(
                job_id=job_id,
                monitor_config=monitor_config,
                metrics_log_path=metrics_log_path
            )
            
            monitor_job.start(self)
            self.active_monitors[job_id] = monitor_job
            logger.info(f"Started monitoring for job {job_id}")
        except Exception as e:
            logger.error(f"Error setting up job monitoring: {str(e)}")
            raise MonitoringError(f"Failed to set up job monitoring: {str(e)}")
    
    def stop_monitoring(self, job_id: str) -> bool:
        """
        Stop monitoring a job.
        
        Args:
            job_id: Job ID to stop monitoring
            
        Returns:
            True if monitoring was stopped
            
        Raises:
            MonitoringError: If monitoring fails to stop
        """
        try:
            if job_id not in self.active_monitors:
                logger.warning(f"Job {job_id} is not being monitored")
                return False
                
            # Stop the monitor
            self.active_monitors[job_id].stop()
            del self.active_monitors[job_id]
            logger.info(f"Stopped monitoring for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            raise MonitoringError(f"Failed to stop monitoring: {str(e)}")
    
    def stop_fine_tune_job(self, job_id: str) -> bool:
        """
        Stop a fine-tuning job and its monitoring.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
            
        Raises:
            JobError: If job operation fails
            MonitoringError: If monitoring operation fails
        """
        try:
            # Stop the job
            result = super().stop_fine_tune_job(job_id)
            
            # Stop monitoring if it exists
            if job_id in self.active_monitors:
                self.stop_monitoring(job_id)
                
            return result
        except Exception as e:
            logger.error(f"Error stopping fine-tuning job: {str(e)}")
            raise JobError(f"Failed to stop fine-tuning job: {str(e)}")
    
    def get_job_status_with_monitoring(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status with monitoring metrics.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict with job status and monitoring metrics
            
        Raises:
            JobError: If job operation fails
            MonitoringError: If monitoring operation fails
        """
        try:
            # Get base job status
            job = super().get_job_status(job_id)
            
            result = job.to_dict()
            
            # Add monitoring data if available
            if job_id in self.active_monitors:
                monitor = self.active_monitors[job_id]
                result["monitoring"] = {
                    "metrics_summary": monitor.monitor.get_metrics_summary(),
                    "issues_detected": monitor.issues_detected,
                    "recovery_actions": monitor.recovery_actions
                }
                
            return result
        except Exception as e:
            logger.error(f"Error getting job status with monitoring: {str(e)}")
            raise JobError(f"Failed to get job status with monitoring: {str(e)}")
    
    def get_monitoring_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get monitoring metrics for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict with monitoring metrics
            
        Raises:
            MonitoringError: If monitoring operation fails
        """
        try:
            if job_id not in self.active_monitors:
                return {"error": "Job is not being monitored"}
                
            monitor = self.active_monitors[job_id]
            return monitor.monitor.get_metrics_summary()
        except Exception as e:
            logger.error(f"Error getting monitoring metrics: {str(e)}")
            raise MonitoringError(f"Failed to get monitoring metrics: {str(e)}")
    
    def start_monitoring_existing_job(self, job_id: str,
                                     monitor_config: Optional[MonitoringConfig] = None,
                                     metrics_log_dir: Optional[str] = None) -> bool:
        """
        Start monitoring an existing job.
        
        Args:
            job_id: Job ID to monitor
            monitor_config: Monitoring configuration
            metrics_log_dir: Directory to save metrics logs
            
        Returns:
            True if monitoring started
            
        Raises:
            JobError: If job operation fails
            MonitoringError: If monitoring operation fails
        """
        try:
            # Verify job exists
            job = super().get_job_status(job_id)
            
            # Skip if already monitoring
            if job_id in self.active_monitors:
                logger.warning(f"Already monitoring job {job_id}")
                return True
                
            # Set up monitoring
            self._setup_job_monitoring(
                job_id, 
                monitor_config or self.default_monitor_config, 
                metrics_log_dir
            )
            
            return True
        except Exception as e:
            logger.error(f"Error starting monitoring for existing job: {str(e)}")
            raise MonitoringError(f"Failed to start monitoring for existing job: {str(e)}")
    
    def create_mini_scale_job(self, config: FineTuneConfig, 
                           num_steps: int = 100,
                           metrics_log_dir: Optional[str] = None) -> Tuple[FineTuneJob, Dict[str, Any]]:
        """
        Create a small-scale rehearsal job to test parameters.
        
        Args:
            config: Fine-tuning configuration
            num_steps: Number of steps to run
            metrics_log_dir: Directory to save metrics logs
            
        Returns:
            Tuple of (job, results)
            
        Raises:
            ValidationError: If configuration is invalid
            JobError: If job operation fails
            MonitoringError: If monitoring operation fails
        """
        try:
            # Validate input
            validate_range(num_steps, "num_steps", 1, 1000)
            
            # Create a copy of the config with minimal steps
            mini_config = FineTuneConfig(**config.dict())
            mini_config.job_name = f"mini-{config.job_name}-{int(time.time())}"
            
            # Limit the number of steps
            mini_config.hyperparameters = dict(config.hyperparameters)
            mini_config.hyperparameters["epochCount"] = "1"
            mini_config.hyperparameters["maxSteps"] = str(num_steps)
            
            # Create monitoring config focused on early detection
            monitor_config = MonitoringConfig(
                enabled=True,
                auto_stop_enabled=True,
                metrics_history_size=10,
                grad_norm_threshold=5.0,     # More sensitive for mini job
                loss_spike_ratio=1.5,        # More sensitive for mini job
                nan_detection_enabled=True
            )
            
            # Start the job with monitoring
            job = self.create_fine_tune_job(
                mini_config,
                enable_monitoring=True,
                monitor_config=monitor_config,
                metrics_log_dir=metrics_log_dir
            )
            
            logger.info(f"Created mini-scale job {job.id} to test parameters")
            
            # Wait for job to complete or fail
            max_wait_time = 60 * 30  # 30 minutes max
            poll_interval = 30  # 30 seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Get latest status
                current_job = self.get_job_status(job.id)
                
                if current_job.status in [FineTuneStatus.COMPLETED, FineTuneStatus.FAILED, FineTuneStatus.STOPPED]:
                    # Job finished, get metrics
                    results = self.get_job_status_with_monitoring(job.id)
                    return job, results
                    
                time.sleep(poll_interval)
            
            # Timeout reached
            logger.warning(f"Mini-scale job {job.id} timed out after {max_wait_time/60} minutes")
            return job, {"error": "Job timed out"}
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating mini-scale job: {str(e)}")
            raise JobError(f"Failed to create mini-scale job: {str(e)}") 