"""
Fine-tuning implementation for AWS Bedrock models.
"""
import boto3
import json
import time
import uuid
from ...config import settings
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class BedrockFineTuner:
    def __init__(self, region_name=None):
        """
        Initialize the Bedrock fine-tuning client.
        
        Args:
            region_name (str, optional): AWS region name, defaults to settings.AWS_REGION
        """
        region = region_name or settings.AWS_REGION
        self.client = boto3.client(
            'bedrock', 
            region_name=region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        logger.info(f"Initialized Bedrock fine-tuning client in region {region}")
        
    def create_fine_tuning_job(self, model_id, training_data_s3_uri, 
                               validation_data_s3_uri=None, hyperparameters=None,
                               custom_name=None):
        """
        Create a model customization (fine-tuning) job.
        
        Args:
            model_id (str): Base model identifier (e.g., anthropic.claude-v2)
            training_data_s3_uri (str): S3 URI for training data
            validation_data_s3_uri (str, optional): S3 URI for validation data
            hyperparameters (dict, optional): Hyperparameters for fine-tuning
            custom_name (str, optional): Custom name for the fine-tuned model
            
        Returns:
            str: The job ARN
        """
        try:
            job_name = custom_name or f"custom-{model_id.replace('.','-')}-{int(time.time())}"
            
            # Set up the data config
            data_config = {
                "trainingDataConfig": {
                    "s3Uri": training_data_s3_uri
                }
            }
            
            if validation_data_s3_uri:
                data_config["validationDataConfig"] = {
                    "s3Uri": validation_data_s3_uri
                }
            
            # Set up hyperparameters with defaults if not provided
            default_hyperparameters = {
                "epochCount": "3",
                "batchSize": "1",
                "learningRate": "0.0005"
            }
            
            final_hyperparameters = hyperparameters or default_hyperparameters
            
            # Create the fine-tuning job
            response = self.client.create_model_customization_job(
                customizationName=job_name,
                baseModelIdentifier=model_id,
                **data_config,
                hyperParameters=final_hyperparameters
            )
            
            job_arn = response['jobArn']
            logger.info(f"Created fine-tuning job {job_name} with ARN: {job_arn}")
            return job_arn
            
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            raise
        
    def check_job_status(self, job_arn):
        """
        Check the status of a fine-tuning job.
        
        Args:
            job_arn (str): The ARN of the fine-tuning job
            
        Returns:
            dict: Job status information
        """
        try:
            response = self.client.get_model_customization_job(
                jobIdentifier=job_arn
            )
            
            status = response.get('status')
            logger.info(f"Fine-tuning job {job_arn} status: {status}")
            
            return {
                "status": status,
                "creation_time": response.get('creationTime'),
                "last_modified_time": response.get('lastModifiedTime'),
                "output_model_arn": response.get('outputModelArn'),
                "failure_message": response.get('failureMessage')
            }
        except Exception as e:
            logger.error(f"Error checking job status for {job_arn}: {str(e)}")
            raise
    
    def list_fine_tuning_jobs(self, max_results=10):
        """
        List all fine-tuning jobs.
        
        Args:
            max_results (int, optional): Maximum number of results to return
            
        Returns:
            list: List of fine-tuning jobs
        """
        try:
            response = self.client.list_model_customization_jobs(
                maxResults=max_results
            )
            
            jobs = response.get('modelCustomizationJobSummaries', [])
            logger.info(f"Retrieved {len(jobs)} fine-tuning jobs")
            
            return jobs
        except Exception as e:
            logger.error(f"Error listing fine-tuning jobs: {str(e)}")
            raise 