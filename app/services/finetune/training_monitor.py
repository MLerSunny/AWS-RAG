"""
Training monitoring system for early divergence detection.

This module provides tools to detect problems early in model training,
allowing us to stop wasteful training jobs and diagnose issues quickly.
"""
import math
import time
import logging
import statistics
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, TypeVar, Generic
from enum import Enum
from pydantic import BaseModel, Field
from collections import deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Union[int, float])

@dataclass
class MetricValue(Generic[T]):
    """Container for a metric value with timestamp."""
    value: T
    timestamp: datetime
    
    def __post_init__(self):
        """Validate the metric value."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(self.value)}")
        if not isinstance(self.timestamp, datetime):
            raise ValueError(f"Timestamp must be datetime, got {type(self.timestamp)}")

class MetricStatus(str, Enum):
    """Status of a monitored metric."""
    NORMAL = "normal"          # Metric is within expected bounds
    WARNING = "warning"        # Metric is outside normal bounds but not critical
    CRITICAL = "critical"      # Metric indicates a serious problem
    UNKNOWN = "unknown"        # Not enough data to determine status

class MetricType(str, Enum):
    """Types of metrics monitored during training."""
    LOSS = "loss"                      # Training loss values
    GRADIENT_NORM = "gradient_norm"    # Gradient L2 norm
    ACTIVATION = "activation"          # Layer activation statistics
    UPDATE_RATIO = "update_ratio"      # Ratio of updates to weights
    VALIDATION = "validation"          # Validation metrics
    RESOURCE = "resource"              # Resource utilization (memory, time)

class MonitoringConfig(BaseModel):
    """Configuration for the training monitor."""
    # General settings
    enabled: bool = True
    metrics_history_size: int = 20
    auto_stop_enabled: bool = False    # Whether to automatically stop training
    
    # Thresholds for alerts and actions
    grad_norm_threshold: float = 10.0  # Max ratio between current and median norm
    loss_spike_ratio: float = 2.0      # Max ratio for loss increase
    update_ratio_threshold: float = 0.1  # Max ratio between updates and weights
    nan_detection_enabled: bool = True  # Detect NaN/Inf values
    
    # Recovery settings
    auto_lr_reduction: bool = False
    lr_reduction_factor: float = 0.2   # How much to reduce LR on issues
    max_recovery_attempts: int = 3     # Maximum attempts to recover
    
    # Validation
    validation_interval_minutes: float = 10.0  # Time-based validation interval
    val_train_ratio_threshold: float = 1.3  # Max ratio of val/train loss

    # Notification
    notification_enabled: bool = False
    dashboard_url: Optional[str] = None


class MetricHistory(Generic[T]):
    """Stores a moving window of metric values with statistical functions."""
    
    def __init__(self, max_size: int = 20):
        """
        Initialize a metric history tracker.
        
        Args:
            max_size: Maximum number of values to keep
            
        Raises:
            ValueError: If max_size is not positive
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.values: deque[MetricValue[T]] = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, value: T, timestamp: Optional[datetime] = None) -> None:
        """
        Add a new value to the history.
        
        Args:
            value: Metric value to add
            timestamp: Optional timestamp for the value
        """
        if not (math.isnan(value) or math.isinf(value)):
            self.values.append(MetricValue(
                value=value,
                timestamp=timestamp or datetime.now()
            ))
    
    def get_values(self, exclude_last: int = 0) -> List[T]:
        """
        Get list of values, optionally excluding recent ones.
        
        Args:
            exclude_last: Number of most recent values to exclude
            
        Returns:
            List of metric values
        """
        values = list(self.values)[:-exclude_last] if exclude_last > 0 else self.values
        return [v.value for v in values]
    
    def mean(self, exclude_last: int = 0) -> Optional[float]:
        """
        Calculate the mean of stored values.
        
        Args:
            exclude_last: Number of most recent values to exclude
            
        Returns:
            Mean value or None if insufficient data
        """
        values = self.get_values(exclude_last)
        if not values:
            return None
        return sum(values) / len(values)
    
    def median(self, exclude_last: int = 0) -> Optional[float]:
        """
        Calculate the median of stored values.
        
        Args:
            exclude_last: Number of most recent values to exclude
            
        Returns:
            Median value or None if insufficient data
        """
        values = self.get_values(exclude_last)
        if not values:
            return None
        return statistics.median(values)
    
    def std_dev(self, exclude_last: int = 0) -> Optional[float]:
        """
        Calculate the standard deviation of stored values.
        
        Args:
            exclude_last: Number of most recent values to exclude
            
        Returns:
            Standard deviation or None if insufficient data
        """
        values = self.get_values(exclude_last)
        if len(values) < 2:
            return None
        return statistics.stdev(values)
    
    def latest(self) -> Optional[T]:
        """Get the most recent value."""
        if not self.values:
            return None
        return self.values[-1].value
    
    def latest_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the most recent value."""
        if not self.values:
            return None
        return self.values[-1].timestamp
    
    def is_empty(self) -> bool:
        """Check if the history is empty."""
        return len(self.values) == 0
    
    def is_full(self) -> bool:
        """Check if the history has reached its maximum size."""
        return len(self.values) == self.max_size
    
    def clear(self) -> None:
        """Clear all stored values."""
        self.values.clear()


class TrainingMonitor:
    """
    Monitors training progress and detects divergence early.
    
    This class tracks various metrics during model training and provides
    early warning of potential issues like divergence, exploding gradients,
    and other training instabilities.
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize the training monitor.
        
        Args:
            config: Monitor configuration
        """
        self.config = config
        self.enabled = config.enabled
        
        # Initialize metric histories
        self.metrics: Dict[str, MetricHistory[float]] = {
            "loss": MetricHistory[float](config.metrics_history_size),
            "grad_norm": MetricHistory[float](config.metrics_history_size),
            "learning_rate": MetricHistory[float](config.metrics_history_size),
        }
        
        # Layer-specific metrics
        self.layer_metrics: Dict[str, Dict[str, MetricHistory[float]]] = {}
        
        # Status tracking
        self.recovery_attempts = 0
        self.last_validation_time = 0
        self.status_history: List[Dict[str, Any]] = []
        
        logger.info(f"Training monitor initialized with config: {config}")
    
    def track_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.LOSS,
                     layer_name: Optional[str] = None) -> None:
        """
        Track a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            layer_name: Name of the layer (for layer-specific metrics)
        """
        if not self.enabled:
            return
            
        # Handle NaN/Inf values
        if math.isnan(value) or math.isinf(value):
            if self.config.nan_detection_enabled:
                logger.warning(f"NaN/Inf detected in metric {name}")
                self._record_status(
                    metric=name,
                    status=MetricStatus.CRITICAL,
                    message=f"NaN/Inf value detected in {name}",
                    metric_type=metric_type
                )
            return
        
        # Track layer-specific metrics
        if layer_name is not None:
            if layer_name not in self.layer_metrics:
                self.layer_metrics[layer_name] = {}
                
            if name not in self.layer_metrics[layer_name]:
                self.layer_metrics[layer_name][name] = MetricHistory[float](self.config.metrics_history_size)
                
            self.layer_metrics[layer_name][name].add(value)
        # Track global metrics
        else:
            if name not in self.metrics:
                self.metrics[name] = MetricHistory[float](self.config.metrics_history_size)
                
            self.metrics[name].add(value)
    
    def check_loss_curve(self) -> Tuple[bool, str, MetricStatus]:
        """
        Check for issues in the loss curve.
        
        Returns:
            Tuple of (is_ok, message, status)
        """
        if "loss" not in self.metrics or self.metrics["loss"].is_empty():
            return True, "Insufficient loss data", MetricStatus.UNKNOWN
            
        current_loss = self.metrics["loss"].latest()
        avg_previous = self.metrics["loss"].mean(exclude_last=1)
        
        if current_loss is None or avg_previous is None:
            return True, "Insufficient loss data", MetricStatus.UNKNOWN
            
        if current_loss > self.config.loss_spike_ratio * avg_previous:
            return False, f"Loss spike detected: {current_loss:.4f} vs avg {avg_previous:.4f}", MetricStatus.CRITICAL
        
        # Check if loss is decreasing overall
        if not self.metrics["loss"].is_empty() and len(self.metrics["loss"].values) >= 10:
            first_half = self.metrics["loss"].get_values()[:len(self.metrics["loss"].values)//2]
            second_half = self.metrics["loss"].get_values()[len(self.metrics["loss"].values)//2:]
            
            first_mean = sum(first_half) / len(first_half)
            second_mean = sum(second_half) / len(second_half)
            
            if second_mean >= first_mean:
                return False, f"Loss not decreasing: {second_mean:.4f} vs {first_mean:.4f}", MetricStatus.WARNING
        
        return True, "Loss curve healthy", MetricStatus.NORMAL
    
    def check_gradient_health(self) -> Tuple[bool, str, MetricStatus]:
        """
        Check for gradient explosion or vanishing.
        
        Returns:
            Tuple of (is_ok, message, status)
        """
        if "grad_norm" not in self.metrics or self.metrics["grad_norm"].is_empty():
            return True, "Insufficient gradient data", MetricStatus.UNKNOWN
            
        current_norm = self.metrics["grad_norm"].latest()
        median_norm = self.metrics["grad_norm"].median()
        
        if current_norm is None or median_norm is None:
            return True, "Insufficient gradient data", MetricStatus.UNKNOWN
            
        # Check for exploding gradients
        if current_norm > self.config.grad_norm_threshold * median_norm:
            return False, f"Gradient explosion: {current_norm:.4f} vs median {median_norm:.4f}", MetricStatus.CRITICAL
        
        # Check for vanishing gradients
        if median_norm > 0 and current_norm < 0.01 * median_norm:
            return False, f"Vanishing gradient: {current_norm:.6f}", MetricStatus.WARNING
            
        return True, "Gradients stable", MetricStatus.NORMAL
    
    def check_activation_statistics(self, layer_name: str) -> Tuple[bool, str, MetricStatus]:
        """
        Check activation statistics for a specific layer.
        
        Args:
            layer_name: Name of the layer to check
            
        Returns:
            Tuple of (is_ok, message, status)
        """
        if layer_name not in self.layer_metrics:
            return True, f"No data for layer {layer_name}", MetricStatus.UNKNOWN
            
        layer_data = self.layer_metrics[layer_name]
        
        if "mean" not in layer_data or "std" not in layer_data:
            return True, f"Insufficient statistics for layer {layer_name}", MetricStatus.UNKNOWN
            
        current_mean = layer_data["mean"].latest()
        current_std = layer_data["std"].latest()
        
        if current_mean is None or current_std is None:
            return True, f"Insufficient statistics for layer {layer_name}", MetricStatus.UNKNOWN
            
        # Check for activation collapse
        if current_std < 0.01:
            return False, f"Activation collapse in {layer_name}: std={current_std:.6f}", MetricStatus.CRITICAL
            
        # Check for exploding activations
        if abs(current_mean) > 10 or current_std > 10:
            return False, f"Activation explosion in {layer_name}: mean={current_mean:.4f}, std={current_std:.4f}", MetricStatus.CRITICAL
            
        return True, f"Layer {layer_name} activations normal", MetricStatus.NORMAL
    
    def check_validation_metrics(self, val_loss: float, train_loss: float) -> Tuple[bool, str, MetricStatus]:
        """
        Check validation metrics against training metrics.
        
        Args:
            val_loss: Current validation loss
            train_loss: Current training loss
            
        Returns:
            Tuple of (is_ok, message, status)
        """
        # Track validation loss
        self.track_metric("val_loss", val_loss, MetricType.VALIDATION)
        
        # Check val/train ratio
        if val_loss > self.config.val_train_ratio_threshold * train_loss:
            return False, f"Validation loss much higher than training: {val_loss:.4f} vs {train_loss:.4f}", MetricStatus.WARNING
            
        # Check if validation loss is increasing
        if "val_loss" in self.metrics and not self.metrics["val_loss"].is_empty():
            prev_val_loss = self.metrics["val_loss"].mean(exclude_last=1)
            if prev_val_loss is not None and val_loss > prev_val_loss:
                return False, f"Validation loss increasing: {val_loss:.4f} vs {prev_val_loss:.4f}", MetricStatus.WARNING
                
        return True, "Validation metrics normal", MetricStatus.NORMAL
    
    def should_run_validation(self) -> bool:
        """
        Determine if validation should be run based on time interval.
        
        Returns:
            True if validation should be run
        """
        current_time = time.time()
        time_since_last = current_time - self.last_validation_time
        
        if time_since_last >= self.config.validation_interval_minutes * 60:
            self.last_validation_time = current_time
            return True
            
        return False
    
    def check_update_ratios(self, update_ratios: List[float]) -> Tuple[bool, str, MetricStatus]:
        """
        Check parameter update to weight ratios.
        
        Args:
            update_ratios: List of update/weight ratios
            
        Returns:
            Tuple of (is_ok, message, status)
        """
        if not update_ratios:
            return True, "No update ratio data", MetricStatus.UNKNOWN
            
        max_ratio = max(update_ratios)
        self.track_metric("update_ratio", max_ratio, MetricType.UPDATE_RATIO)
        
        if max_ratio > self.config.update_ratio_threshold:
            return False, f"Update/weight ratio too high: {max_ratio:.4f}", MetricStatus.CRITICAL
            
        return True, "Update ratios normal", MetricStatus.NORMAL
    
    def should_stop_training(self) -> Tuple[bool, str]:
        """
        Determine if training should be stopped due to issues.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if not self.config.auto_stop_enabled:
            return False, "Auto-stop disabled"
            
        # Check recent status history
        critical_count = sum(1 for entry in self.status_history[-5:] 
                          if entry["status"] == MetricStatus.CRITICAL)
                          
        if critical_count >= 3:
            return True, "Multiple critical issues detected"
            
        # Check if recovery attempts exhausted
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            return True, f"Recovery attempts exhausted ({self.recovery_attempts})"
            
        return False, "No stop condition met"
    
    def should_reduce_learning_rate(self) -> bool:
        """
        Determine if learning rate should be reduced due to issues.
        
        Returns:
            True if LR should be reduced
        """
        if not self.config.auto_lr_reduction:
            return False
            
        # Check if we've had recent critical issues
        critical_issues = any(entry["status"] == MetricStatus.CRITICAL 
                             for entry in self.status_history[-3:])
                             
        # Only attempt recovery if we haven't exceeded the limit
        if critical_issues and self.recovery_attempts < self.config.max_recovery_attempts:
            self.recovery_attempts += 1
            return True
            
        return False
    
    def get_reduction_factor(self) -> float:
        """
        Get the learning rate reduction factor.
        
        Returns:
            Factor to multiply learning rate by
        """
        return self.config.lr_reduction_factor
    
    def _record_status(self, metric: str, status: MetricStatus, 
                      message: str, metric_type: MetricType) -> None:
        """
        Record a status entry in the history.
        
        Args:
            metric: Name of the metric
            status: Status value
            message: Status message
            metric_type: Type of metric
        """
        entry = {
            "timestamp": time.time(),
            "metric": metric,
            "status": status,
            "message": message,
            "type": metric_type
        }
        
        self.status_history.append(entry)
        
        # Log based on severity
        if status == MetricStatus.CRITICAL:
            logger.error(message)
        elif status == MetricStatus.WARNING:
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        summary = {
            "global_metrics": {},
            "layer_metrics": {},
            "status": self._get_overall_status(),
            "recovery_attempts": self.recovery_attempts
        }
        
        # Add global metrics
        for name, history in self.metrics.items():
            if not history.is_empty():
                summary["global_metrics"][name] = {
                    "current": history.latest(),
                    "mean": history.mean(),
                    "median": history.median(),
                    "std_dev": history.std_dev() 
                }
        
        # Add layer metrics
        for layer_name, metrics in self.layer_metrics.items():
            summary["layer_metrics"][layer_name] = {}
            for name, history in metrics.items():
                if not history.is_empty():
                    summary["layer_metrics"][layer_name][name] = {
                        "current": history.latest(),
                        "mean": history.mean()
                    }
        
        return summary
    
    def _get_overall_status(self) -> MetricStatus:
        """
        Get overall status based on recent history.
        
        Returns:
            Overall status
        """
        if not self.status_history:
            return MetricStatus.UNKNOWN
            
        # Check last 5 entries
        recent = self.status_history[-5:]
        
        if any(entry["status"] == MetricStatus.CRITICAL for entry in recent):
            return MetricStatus.CRITICAL
            
        if any(entry["status"] == MetricStatus.WARNING for entry in recent):
            return MetricStatus.WARNING
            
        return MetricStatus.NORMAL 