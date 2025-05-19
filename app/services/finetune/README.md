# AWS Bedrock Fine-Tuning with Early Divergence Detection

This module provides enhanced AWS Bedrock fine-tuning capabilities with early divergence detection to save time and resources by catching training issues quickly.

## Features

- **Early Divergence Detection**: Automatically detect and respond to training issues within minutes, not hours
- **Fine-Tuning Monitoring**: Real-time tracking of loss curves, validation metrics, and other training indicators
- **Mini-Scale Rehearsal**: Test fine-tuning configurations on a small scale before committing to full training
- **Hyperparameter Validation**: Detect common issues in hyperparameter configurations before starting training
- **Smart Hyperparameter Suggestions**: Get optimal hyperparameter recommendations based on model and dataset properties
- **Time-Based Validation**: Run validation at regular time intervals rather than step intervals

## Metrics Monitored

| Metric Type | Description | Detection Mechanism |
|-------------|-------------|---------------------|
| Loss Curve | Training loss values | Detects spikes, stagnation, oscillations |
| Validation Loss | Validation set performance | Compares against training loss; checks for increasing trend |
| Gradient Norms (estimated) | Magnitude of gradients | Estimates based on loss changes and learning rate |
| Resource Usage | Step time, memory | Watches for unexpected resource consumption patterns |

## Early Warning Signals

The system can detect the following early warning signals:

1. **NaN/Inf Loss**: Immediately detects numerical instability issues
2. **Loss Spikes**: Detects sudden increases in loss value that indicate instability
3. **Training Stagnation**: Detects when loss stops decreasing over time
4. **Validation/Training Divergence**: Warns when validation loss rises while training loss decreases
5. **Resource Anomalies**: Detects unusual step times or memory usage

## CLI Usage

The module includes a command-line interface for using these features:

```bash
# Run a small-scale rehearsal job
python -m app.services.finetune.cli mini-run --config config.json --steps 100

# Run full fine-tuning with monitoring
python -m app.services.finetune.cli run --config config.json --auto-stop

# Monitor an existing job
python -m app.services.finetune.cli monitor --job-id YOUR_JOB_ID

# Validate hyperparameters
python -m app.services.finetune.cli validate --config config.json

# Get optimal hyperparameters
python -m app.services.finetune.cli get-optimal --model-type anthropic.claude-v2 --dataset-size 1000
```

## Configuration

Sample configuration file (config.json):

```json
{
  "job_name": "my-finetune-job",
  "base_model_id": "anthropic.claude-v2",
  "training_data_path": "s3://my-bucket/training.jsonl",
  "output_data_path": "s3://my-bucket/output/",
  "hyperparameters": {
    "batchSize": "2",
    "epochCount": "3",
    "learningRate": "0.0001",
    "weightDecay": "0.01",
    "warmupSteps": "100",
    "scheduler": "cosine"
  },
  "enable_peft": true,
  "peft_method": "lora",
  "lora_rank": 8
}
```

## Automatic Recovery

The system can be configured to automatically respond to issues:

1. **Auto-Stop**: Automatically stop training when critical issues are detected
2. **Metrics Logging**: Save detailed metrics to disk for post-training analysis
3. **Alerting**: Record all issues detected for review

## Implementation Notes

- The monitoring runs in a separate thread to avoid blocking the main execution
- Metrics are saved to disk at regular intervals for audit and analysis
- Recovery actions are logged alongside issues
- All divergence detection is non-intrusive to the AWS Bedrock service

## Prerequisites

- AWS Bedrock access with fine-tuning permissions
- S3 bucket for training data and output models
- AWS credentials configured in environment

## Best Practices

1. **Always run a mini-job first**: Use the `mini-run` command to test configurations
2. **Enable auto-stop in production**: Set `--auto-stop` to prevent wasting resources on failing jobs
3. **Validate hyperparameters**: Check for issues before starting expensive training
4. **Monitor metrics logs**: Review the metrics logs even for successful jobs to optimize future runs
5. **Use optimal hyperparameters**: Generate suggested parameters based on your specific use case

## Monitoring Dashboard

For visualization of metrics, point the metrics logs directory to a monitoring solution, or implement a custom dashboard using the generated JSON logs.

## Architecture

The system includes the following components:

1. `EnhancedBedrockFineTuner`: Extends the base Bedrock integration with monitoring
2. `TrainingMonitor`: Core monitoring and divergence detection logic
3. `BedrockMonitoringJob`: Background job that polls for status and metrics
4. `training_utils`: Helper functions for parameter validation and optimization

## Limitations

- AWS Bedrock doesn't provide direct access to gradients, so we use approximations
- Cannot dynamically adjust learning rate in AWS Bedrock fine-tuning
- Resource monitoring is limited to metrics exposed by AWS Bedrock

## Future Improvements

- Integration with CloudWatch alarms and notifications
- Automatic hyperparameter optimization based on mini-job results
- Support for custom monitoring thresholds by model type 