"""
Training utilities for fine-tuning models.

This module provides utility functions for monitoring training,
estimating optimal parameters, and detecting issues early.
"""
import math
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from ..training_monitor import MonitoringConfig, MetricStatus
from ....utils.logger import setup_logger

logger = setup_logger(__name__)

def estimate_warmup_steps(dataset_size: int, batch_size: int, 
                          num_epochs: int, warmup_ratio: float = 0.1) -> int:
    """
    Estimate the optimal number of warmup steps.
    
    Args:
        dataset_size: Number of examples in the dataset
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        warmup_ratio: Ratio of training to use for warmup
        
    Returns:
        Number of warmup steps
    """
    # Calculate total training steps
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    # Apply warmup ratio
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    
    return warmup_steps

def estimate_learning_rate(model_type: str, dataset_size: int,
                          fine_tuning_type: str = "standard") -> float:
    """
    Estimate a good starting learning rate.
    
    Args:
        model_type: Type of base model
        dataset_size: Number of examples in the dataset
        fine_tuning_type: Type of fine-tuning (standard or peft)
        
    Returns:
        Recommended learning rate
    """
    # Base learning rates by model type
    lr_base = {
        "anthropic.claude": 1e-5,
        "amazon.titan": 2e-5,
        "meta.llama": 3e-5,
        "mistral.mistral": 2e-5,
        "default": 1e-4
    }
    
    # Find the most specific model match
    model_lr = lr_base.get("default")
    for model_prefix, lr in lr_base.items():
        if model_type.startswith(model_prefix):
            model_lr = lr
            break
    
    # Ensure model_lr is not None before applying multipliers
    if model_lr is None:
        model_lr = 1e-4  # Fallback default
    
    # Adjust based on dataset size
    if dataset_size < 100:
        # Small datasets need lower learning rates to avoid overfitting
        model_lr = model_lr * 0.5
    elif dataset_size > 10000:
        # Larger datasets can use slightly higher learning rates
        model_lr = model_lr * 1.5
    
    # Adjust based on fine-tuning type
    if fine_tuning_type == "peft" or fine_tuning_type == "lora":
        # LoRA typically uses higher learning rates
        model_lr = model_lr * 3.0
    
    return model_lr

def validate_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix hyperparameters to avoid common issues.
    
    Args:
        config: Hyperparameter configuration
        
    Returns:
        Dict with issues and suggestions
    """
    issues = []
    warnings = []
    suggestions = {}
    
    # Convert string params to appropriate types for validation
    params = {}
    for k, v in config.items():
        try:
            # Convert numeric strings to numbers
            if isinstance(v, str) and v.replace('.', '', 1).isdigit():
                if '.' in v:
                    params[k] = float(v)
                else:
                    params[k] = int(v)
            else:
                params[k] = v
        except:
            params[k] = v
    
    # Check learning rate
    if "learningRate" in params:
        lr = params["learningRate"]
        if lr > 0.1:
            issues.append(f"Learning rate ({lr}) is suspiciously high")
            suggestions["learningRate"] = "1e-4"
        elif lr < 1e-7:
            issues.append(f"Learning rate ({lr}) is extremely low")
            suggestions["learningRate"] = "1e-5"
    else:
        warnings.append("Learning rate not specified")
        suggestions["learningRate"] = "2e-5"
    
    # Check batch size
    if "batchSize" in params:
        bs = params["batchSize"]
        if bs < 1:
            issues.append(f"Batch size ({bs}) must be at least 1")
            suggestions["batchSize"] = "1"
        elif bs > 64:
            warnings.append(f"Batch size ({bs}) is very large, may cause memory issues")
    else:
        warnings.append("Batch size not specified")
        suggestions["batchSize"] = "1"
    
    # Check epoch count
    if "epochCount" in params:
        epochs = params["epochCount"]
        if epochs < 1:
            issues.append(f"Epoch count ({epochs}) must be at least 1")
            suggestions["epochCount"] = "3"
        elif epochs > 10:
            warnings.append(f"Epoch count ({epochs}) is high, consider early stopping")
    else:
        warnings.append("Epoch count not specified")
        suggestions["epochCount"] = "3"
    
    # Check warmup steps
    if "warmupSteps" in params:
        warmup = params["warmupSteps"]
        if warmup < 0:
            issues.append(f"Warmup steps ({warmup}) cannot be negative")
            suggestions["warmupSteps"] = "100"
    else:
        warnings.append("Warmup steps not specified")
        suggestions["warmupSteps"] = "100"
    
    # Check weight decay
    if "weightDecay" in params:
        wd = params["weightDecay"]
        if wd < 0:
            issues.append(f"Weight decay ({wd}) cannot be negative")
            suggestions["weightDecay"] = "0.01"
        elif wd > 0.1:
            warnings.append(f"Weight decay ({wd}) is high, may slow convergence")
    
    # Check for potential conflicts or combinations that cause issues
    if "learningRate" in params and "warmupSteps" in params:
        lr = params["learningRate"]
        warmup = params["warmupSteps"]
        
        if lr > 1e-3 and warmup < 10:
            warnings.append(f"High learning rate ({lr}) with few warmup steps ({warmup}) may cause instability")
            suggestions["warmupSteps"] = "200"
    
    return {
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions
    }

def compute_gradient_norm_estimate(loss_values: List[float], 
                                 learning_rate: float,
                                 consecutive_steps: int = 3) -> Optional[float]:
    """
    Estimate gradient norm using loss changes and learning rate.
    
    This is an approximation when direct gradient access is unavailable.
    
    Args:
        loss_values: Recent loss values
        learning_rate: Current learning rate
        consecutive_steps: Number of consecutive steps to consider
        
    Returns:
        Estimated gradient norm or None if insufficient data
    """
    if len(loss_values) < consecutive_steps + 1:
        return None
    
    # Take the most recent steps
    recent_losses = loss_values[-consecutive_steps-1:]
    
    # Calculate average loss change per step
    loss_diffs = [abs(recent_losses[i+1] - recent_losses[i]) for i in range(consecutive_steps)]
    avg_loss_change = sum(loss_diffs) / consecutive_steps
    
    # Estimate gradient norm using the relationship: loss_change ≈ learning_rate * gradient_norm
    # This is a crude approximation based on first-order optimization
    if learning_rate > 0:
        estimated_grad_norm = avg_loss_change / learning_rate
        return float(estimated_grad_norm)
    
    return None

def analyze_loss_curve(loss_values: List[float]) -> Dict[str, Any]:
    """
    Analyze a loss curve for potential issues.
    
    Args:
        loss_values: List of loss values over time
        
    Returns:
        Dict with analysis results
    """
    if len(loss_values) < 5:
        return {"status": "insufficient_data", "message": "Need more data points"}
    
    result = {
        "min_loss": min(loss_values),
        "max_loss": max(loss_values),
        "latest_loss": loss_values[-1],
        "issues": []
    }
    
    # Check for NaN or infinity
    if any(math.isnan(loss) or math.isinf(loss) for loss in loss_values):
        result["issues"].append({
            "type": "numerical_instability",
            "message": "Loss contains NaN or infinite values",
            "severity": "critical"
        })
        result["status"] = MetricStatus.CRITICAL
        return result
    
    # Calculate loss changes
    loss_diffs = [loss_values[i+1] - loss_values[i] for i in range(len(loss_values)-1)]
    
    # Check for loss spikes
    for i, diff in enumerate(loss_diffs):
        if diff > 0 and diff > loss_values[i] * 0.5:  # Loss increased by more than 50%
            result["issues"].append({
                "type": "loss_spike",
                "message": f"Loss spike detected at step {i+1}: {loss_values[i]:.4f} -> {loss_values[i+1]:.4f}",
                "severity": "warning",
                "position": i+1
            })
    
    # Check if loss is decreasing overall
    if len(loss_values) >= 10:
        first_half = loss_values[:len(loss_values)//2]
        second_half = loss_values[len(loss_values)//2:]
        
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        
        if second_mean >= first_mean:
            result["issues"].append({
                "type": "stagnation",
                "message": f"Loss not decreasing overall: {first_mean:.4f} -> {second_mean:.4f}",
                "severity": "warning"
            })
    
    # Check for oscillation
    if len(loss_diffs) >= 6:
        sign_changes = sum(1 for i in range(len(loss_diffs)-1) if loss_diffs[i] * loss_diffs[i+1] < 0)
        oscillation_ratio = sign_changes / (len(loss_diffs) - 1)
        
        if oscillation_ratio > 0.5:
            result["issues"].append({
                "type": "oscillation",
                "message": f"Loss is oscillating (sign changes in {oscillation_ratio:.1%} of steps)",
                "severity": "warning"
            })
    
    # Determine overall status
    critical_issues = [issue for issue in result["issues"] if issue["severity"] == "critical"]
    warning_issues = [issue for issue in result["issues"] if issue["severity"] == "warning"]
    
    if critical_issues:
        result["status"] = MetricStatus.CRITICAL
    elif warning_issues:
        result["status"] = MetricStatus.WARNING
    else:
        result["status"] = MetricStatus.NORMAL
    
    return result

def create_default_monitoring_config(
    training_type: str = "standard",
    model_size_b: float = 7.0,
    auto_stop: bool = False
) -> MonitoringConfig:
    """
    Create a default monitoring configuration based on training parameters.
    
    Args:
        training_type: Type of training (standard, lora, etc.)
        model_size_b: Size of model in billions of parameters
        auto_stop: Whether to enable automatic stopping
        
    Returns:
        MonitoringConfig instance
    """
    config = MonitoringConfig(
        enabled=True,
        auto_stop_enabled=auto_stop,
        metrics_history_size=20,
        grad_norm_threshold=10.0,
        loss_spike_ratio=2.0,
        update_ratio_threshold=0.1,
        auto_lr_reduction=False,  # AWS Bedrock doesn't support dynamic LR changes
        validation_interval_minutes=10.0
    )
    
    # Adjust thresholds based on model size
    if model_size_b > 20:
        # Larger models often need more careful monitoring
        config.grad_norm_threshold = 5.0
        config.loss_spike_ratio = 1.5
        config.update_ratio_threshold = 0.05
        config.validation_interval_minutes = 15.0
    
    # Adjust based on training type
    if training_type == "lora" or training_type == "peft":
        # LoRA training can tolerate higher gradient norms
        config.grad_norm_threshold = 15.0
        config.update_ratio_threshold = 0.2
    
    return config

def calculate_optimal_hyperparameters(
    model_type: str,
    dataset_size: int,
    training_type: str = "standard",
    target_hardware: str = "default",
    time_constraint_hours: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate optimal hyperparameters for fine-tuning.
    
    Args:
        model_type: Type of base model
        dataset_size: Number of examples in the dataset
        training_type: Type of fine-tuning (standard, lora, etc.)
        target_hardware: Hardware target (helps with batch size estimation)
        time_constraint_hours: Time constraint in hours
        
    Returns:
        Dict of recommended hyperparameters
    """
    result = {
        "learningRate": estimate_learning_rate(model_type, dataset_size, training_type),
        "weightDecay": 0.01,
        "scheduler": "cosine"
    }
    
    # Determine batch size based on hardware and model
    if target_hardware == "high_memory":
        base_batch_size = 8
    elif target_hardware == "limited_memory":
        base_batch_size = 1
    else:
        base_batch_size = 4
        
    # Adjust batch size for LoRA
    if training_type == "lora" or training_type == "peft":
        base_batch_size *= 2
    
    result["batchSize"] = base_batch_size
    
    # Calculate epochs based on dataset size
    if dataset_size < 100:
        # Small datasets need more epochs
        result["epochCount"] = 8
    elif dataset_size < 1000:
        result["epochCount"] = 5
    elif dataset_size < 10000:
        result["epochCount"] = 3
    else:
        # Large datasets can use fewer epochs
        result["epochCount"] = 2
    
    # Apply time constraint if specified
    if time_constraint_hours is not None and time_constraint_hours > 0:
        # Rough estimate: adjust epochs to fit in time constraint
        # Assuming 1000 examples per hour as baseline
        estimated_examples_per_hour = 1000 * base_batch_size
        max_epochs = (time_constraint_hours * estimated_examples_per_hour) / dataset_size
        
        if max_epochs < result["epochCount"]:
            result["epochCount"] = max(1, round(max_epochs))
            logger.info(f"Reduced epochs to {result['epochCount']} due to time constraint")
    
    # Calculate warmup steps (10% of training by default)
    steps_per_epoch = dataset_size // base_batch_size
    result["warmupSteps"] = max(100, int(0.1 * steps_per_epoch * result["epochCount"]))
    
    # Convert all values to strings for Bedrock API
    for key in result:
        result[key] = str(result[key])
    
    return result 