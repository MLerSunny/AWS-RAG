"""
Command-line interface for enhanced Bedrock fine-tuning with monitoring.

This module provides CLI commands to run fine-tuning jobs with early
divergence detection and monitoring.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
import logging

from .bedrock_finetune_monitor import EnhancedBedrockFineTuner
from .training_monitor import MonitoringConfig, MetricStatus
from .utils.training_utils import (
    validate_hyperparameters, 
    create_default_monitoring_config,
    calculate_optimal_hyperparameters
)
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced AWS Bedrock fine-tuning with early divergence detection"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # COMMAND: Run mini-scale job
    mini_parser = subparsers.add_parser(
        "mini-run", help="Run a small-scale job to test parameters before full training"
    )
    mini_parser.add_argument(
        "--config", type=str, required=True,
        help="JSON config file for fine-tuning"
    )
    mini_parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of steps to run in mini job"
    )
    mini_parser.add_argument(
        "--output-dir", type=str, default="./finetune_logs",
        help="Directory to save metrics logs"
    )
    
    # COMMAND: Run with monitoring
    run_parser = subparsers.add_parser(
        "run", help="Run a fine-tuning job with monitoring"
    )
    run_parser.add_argument(
        "--config", type=str, required=True,
        help="JSON config file for fine-tuning"
    )
    run_parser.add_argument(
        "--auto-stop", action="store_true",
        help="Enable auto-stopping on detected issues"
    )
    run_parser.add_argument(
        "--output-dir", type=str, default="./finetune_logs",
        help="Directory to save metrics logs"
    )
    run_parser.add_argument(
        "--model-size", type=float, default=7.0,
        help="Model size in billions of parameters (affects monitoring thresholds)"
    )
    
    # COMMAND: Monitor existing job
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor an existing fine-tuning job"
    )
    monitor_parser.add_argument(
        "--job-id", type=str, required=True,
        help="ID of existing fine-tuning job"
    )
    monitor_parser.add_argument(
        "--auto-stop", action="store_true",
        help="Enable auto-stopping on detected issues"
    )
    monitor_parser.add_argument(
        "--output-dir", type=str, default="./finetune_logs",
        help="Directory to save metrics logs"
    )
    
    # COMMAND: Validate hyperparameters
    validate_parser = subparsers.add_parser(
        "validate", help="Validate hyperparameters for potential issues"
    )
    validate_parser.add_argument(
        "--config", type=str, required=True,
        help="JSON config file for fine-tuning"
    )
    
    # COMMAND: Get optimal hyperparameters
    optimal_parser = subparsers.add_parser(
        "get-optimal", help="Generate optimal hyperparameters"
    )
    optimal_parser.add_argument(
        "--model-type", type=str, required=True,
        help="Base model type (e.g., anthropic.claude-v2)"
    )
    optimal_parser.add_argument(
        "--dataset-size", type=int, required=True,
        help="Number of examples in the training dataset"
    )
    optimal_parser.add_argument(
        "--training-type", type=str, default="standard", 
        choices=["standard", "lora", "peft"],
        help="Type of fine-tuning to perform"
    )
    optimal_parser.add_argument(
        "--hardware", type=str, default="default",
        choices=["default", "high_memory", "limited_memory"],
        help="Target hardware profile"
    )
    optimal_parser.add_argument(
        "--time-constraint", type=float, default=None,
        help="Time constraint in hours (optional)"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from a JSON file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dict containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Basic validation
        required_fields = ["job_name", "base_model_id", "training_data_path", "output_data_path"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in config: {field}")
                sys.exit(1)
                
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        sys.exit(1)

def run_mini_scale_job(args):
    """
    Run a small-scale job to test parameters.
    
    Args:
        args: Command-line arguments
    """
    config_data = load_config(args.config)
    
    # Create fine-tuner
    finetuner = EnhancedBedrockFineTuner()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run mini-scale job
    from .bedrock_finetune import FineTuneConfig
    config = FineTuneConfig(**config_data)
    
    logger.info(f"Running mini-scale job with {args.steps} steps to test parameters")
    job, results = finetuner.create_mini_scale_job(
        config=config,
        num_steps=args.steps,
        metrics_log_dir=args.output_dir
    )
    
    if job is None:
        logger.error("Failed to create mini-scale job")
        sys.exit(1)
    
    # Save results
    result_path = os.path.join(args.output_dir, f"mini_job_{job.id}_results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info(f"Mini-scale job {job.id} completed")
    logger.info(f"Status: {job.status}")
    
    if "recommendation" in results:
        logger.info(f"Recommendation: {results['recommendation']}")
    
    logger.info(f"Results saved to {result_path}")

def run_with_monitoring(args):
    """
    Run a fine-tuning job with monitoring.
    
    Args:
        args: Command-line arguments
    """
    config_data = load_config(args.config)
    
    # Create fine-tuner
    finetuner = EnhancedBedrockFineTuner()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate hyperparameters
    if "hyperparameters" in config_data:
        validation = validate_hyperparameters(config_data["hyperparameters"])
        if validation["issues"]:
            logger.warning("Found issues with hyperparameters:")
            for issue in validation["issues"]:
                logger.warning(f"  - {issue}")
            
            # Apply suggested fixes if critical issues
            if validation["suggestions"]:
                logger.info("Applying suggested fixes:")
                for param, value in validation["suggestions"].items():
                    if param in config_data["hyperparameters"] and param in validation["suggestions"]:
                        logger.info(f"  - Changing {param}: {config_data['hyperparameters'][param]} -> {value}")
                        config_data["hyperparameters"][param] = value
    
    # Create monitoring config
    training_type = "standard"
    if config_data.get("enable_peft", False):
        training_type = config_data.get("peft_method", "lora")
        
    monitor_config = create_default_monitoring_config(
        training_type=training_type,
        model_size_b=args.model_size,
        auto_stop=args.auto_stop
    )
    
    # Run job with monitoring
    from .bedrock_finetune import FineTuneConfig
    config = FineTuneConfig(**config_data)
    
    logger.info(f"Running fine-tuning job with monitoring")
    job = finetuner.create_fine_tune_job(
        config=config,
        enable_monitoring=True,
        monitor_config=monitor_config,
        metrics_log_dir=args.output_dir
    )
    
    if job is None:
        logger.error("Failed to create fine-tuning job")
        sys.exit(1)
    
    # Print job info
    logger.info(f"Created fine-tuning job {job.id}")
    logger.info(f"Status: {job.status}")
    logger.info(f"Monitoring logs will be saved to {args.output_dir}")
    logger.info(f"Auto-stop: {'Enabled' if args.auto_stop else 'Disabled'}")
    logger.info("Job is running in the background. You can check status with AWS console or Bedrock API.")

def monitor_existing_job(args):
    """
    Monitor an existing fine-tuning job.
    
    Args:
        args: Command-line arguments
    """
    # Create fine-tuner
    finetuner = EnhancedBedrockFineTuner()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create monitoring config
    monitor_config = create_default_monitoring_config(
        auto_stop=args.auto_stop
    )
    
    # Start monitoring
    success = finetuner.start_monitoring_existing_job(
        job_id=args.job_id,
        monitor_config=monitor_config,
        metrics_log_dir=args.output_dir
    )
    
    if not success:
        logger.error(f"Failed to monitor job {args.job_id}")
        sys.exit(1)
    
    logger.info(f"Started monitoring for job {args.job_id}")
    logger.info(f"Monitoring logs will be saved to {args.output_dir}")
    logger.info(f"Auto-stop: {'Enabled' if args.auto_stop else 'Disabled'}")
    logger.info("Monitoring is running in the background. Press Ctrl+C to stop the program, monitoring will continue.")
    
    # Keep the program running
    try:
        while True:
            time.sleep(10)
            status = finetuner.get_job_status_with_monitoring(args.job_id)
            if "status" in status and status["status"] in ["completed", "failed", "stopped"]:
                logger.info(f"Job {args.job_id} finished with status: {status['status']}")
                break
    except KeyboardInterrupt:
        logger.info("Exiting program, monitoring will continue in the background")

def validate_hyperparams(args):
    """
    Validate hyperparameters from a config file.
    
    Args:
        args: Command-line arguments
    """
    config_data = load_config(args.config)
    
    if "hyperparameters" not in config_data:
        logger.error("No hyperparameters found in config file")
        sys.exit(1)
    
    validation = validate_hyperparameters(config_data["hyperparameters"])
    
    # Print results
    print("\n===== Hyperparameter Validation Results =====")
    
    if validation["issues"]:
        print("\nIssues (should fix):")
        for issue in validation["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo critical issues found.")
    
    if validation["warnings"]:
        print("\nWarnings (consider fixing):")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    else:
        print("\nNo warnings found.")
    
    if validation["suggestions"]:
        print("\nSuggested fixes:")
        for param, value in validation["suggestions"].items():
            print(f"  - {param}: change to {value}")
    
    print("\nCurrent hyperparameters:")
    for key, value in config_data["hyperparameters"].items():
        print(f"  - {key}: {value}")
    
    print("\n============================================")

def get_optimal_hyperparams(args):
    """
    Generate optimal hyperparameters.
    
    Args:
        args: Command-line arguments
    """
    optimal = calculate_optimal_hyperparameters(
        model_type=args.model_type,
        dataset_size=args.dataset_size,
        training_type=args.training_type,
        target_hardware=args.hardware,
        time_constraint_hours=args.time_constraint
    )
    
    # Print results
    print("\n===== Optimal Hyperparameters =====")
    print(f"\nModel: {args.model_type}")
    print(f"Dataset size: {args.dataset_size} examples")
    print(f"Training type: {args.training_type}")
    print(f"Hardware target: {args.hardware}")
    
    if args.time_constraint:
        print(f"Time constraint: {args.time_constraint} hours")
    
    print("\nRecommended hyperparameters:")
    for key, value in optimal.items():
        print(f"  - {key}: {value}")
    
    print("\nJSON format (for config file):")
    print(json.dumps(optimal, indent=2))
    
    print("\n==================================")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Execute the specified command
    if args.command == "mini-run":
        run_mini_scale_job(args)
    elif args.command == "run":
        run_with_monitoring(args)
    elif args.command == "monitor":
        monitor_existing_job(args)
    elif args.command == "validate":
        validate_hyperparams(args)
    elif args.command == "get-optimal":
        get_optimal_hyperparams(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 