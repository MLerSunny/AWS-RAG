"""
DynamoDB service for persistent storage of application state.
"""
import boto3
import time
import os
import json
from typing import Dict, List, Optional, Any, Union
from botocore.exceptions import ClientError
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class DynamoDBService:
    """Service for interacting with DynamoDB tables."""
    
    def __init__(self, table_prefix: str = "genai_"):
        """
        Initialize the DynamoDB service.
        
        Args:
            table_prefix (str): Prefix for DynamoDB table names
        """
        # Use local endpoint for local development if specified
        endpoint_url = os.environ.get("DYNAMODB_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using local DynamoDB endpoint: {endpoint_url}")
            self.dynamodb = boto3.resource('dynamodb', endpoint_url=endpoint_url)
        else:
            self.dynamodb = boto3.resource('dynamodb')
            
        self.table_prefix = table_prefix
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required DynamoDB tables exist."""
        tables = list(self.dynamodb.tables.all())
        table_names = [table.name for table in tables]
        
        # Model registry table
        model_table_name = f"{self.table_prefix}models"
        if model_table_name not in table_names:
            try:
                logger.info(f"Creating model registry table: {model_table_name}")
                self.dynamodb.create_table(
                    TableName=model_table_name,
                    KeySchema=[
                        {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'id', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                # Wait for table creation
                table = self.dynamodb.Table(model_table_name)
                table.meta.client.get_waiter('table_exists').wait(TableName=model_table_name)
                logger.info(f"Created model registry table: {model_table_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceInUseException':
                    logger.info(f"Table {model_table_name} already exists")
                else:
                    logger.error(f"Error creating table {model_table_name}: {e}")
                    raise
        
        # Model artifacts table
        artifacts_table_name = f"{self.table_prefix}model_artifacts"
        if artifacts_table_name not in table_names:
            try:
                logger.info(f"Creating model artifacts table: {artifacts_table_name}")
                self.dynamodb.create_table(
                    TableName=artifacts_table_name,
                    KeySchema=[
                        {'AttributeName': 'model_id', 'KeyType': 'HASH'},  # Partition key
                        {'AttributeName': 'version', 'KeyType': 'RANGE'}   # Sort key
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'model_id', 'AttributeType': 'S'},
                        {'AttributeName': 'version', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                # Wait for table creation
                table = self.dynamodb.Table(artifacts_table_name)
                table.meta.client.get_waiter('table_exists').wait(TableName=artifacts_table_name)
                logger.info(f"Created model artifacts table: {artifacts_table_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceInUseException':
                    logger.info(f"Table {artifacts_table_name} already exists")
                else:
                    logger.error(f"Error creating table {artifacts_table_name}: {e}")
                    raise
        
        # Model metrics table
        metrics_table_name = f"{self.table_prefix}model_metrics"
        if metrics_table_name not in table_names:
            try:
                logger.info(f"Creating model metrics table: {metrics_table_name}")
                self.dynamodb.create_table(
                    TableName=metrics_table_name,
                    KeySchema=[
                        {'AttributeName': 'model_id', 'KeyType': 'HASH'},  # Partition key
                        {'AttributeName': 'metric_id', 'KeyType': 'RANGE'}  # Sort key
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'model_id', 'AttributeType': 'S'},
                        {'AttributeName': 'metric_id', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                # Wait for table creation
                table = self.dynamodb.Table(metrics_table_name)
                table.meta.client.get_waiter('table_exists').wait(TableName=metrics_table_name)
                logger.info(f"Created model metrics table: {metrics_table_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceInUseException':
                    logger.info(f"Table {metrics_table_name} already exists")
                else:
                    logger.error(f"Error creating table {metrics_table_name}: {e}")
                    raise

    def save_model(self, model_id: str, model_data: Dict[str, Any]) -> bool:
        """
        Save a model to the model registry.
        
        Args:
            model_id (str): Unique identifier for the model
            model_data (Dict[str, Any]): Model configuration data
            
        Returns:
            bool: True if the operation was successful
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}models")
            
            # Add timestamps if not present
            if 'created_at' not in model_data:
                model_data['created_at'] = int(time.time())
            
            model_data['updated_at'] = int(time.time())
            
            # Add the ID to the item
            item = {
                'id': model_id,
                **model_data
            }
            
            table.put_item(Item=item)
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {str(e)}")
            return False
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model from the model registry.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            Optional[Dict[str, Any]]: Model configuration or None if not found
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}models")
            response = table.get_item(Key={'id': model_id})
            
            if 'Item' in response:
                return response['Item']
            return None
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {str(e)}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of model configurations
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}models")
            response = table.scan()
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a model in the registry.
        
        Args:
            model_id (str): Unique identifier for the model
            updates (Dict[str, Any]): Fields to update
            
        Returns:
            bool: True if the update was successful
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}models")
            
            # Get the existing model to check it exists
            existing = self.get_model(model_id)
            if not existing:
                return False
            
            # Build the update expression
            update_expression = "SET updated_at = :updated_at"
            expression_attribute_values = {
                ':updated_at': int(time.time())
            }
            
            # Add each update to the expression
            for key, value in updates.items():
                if key not in ('id', 'created_at', 'updated_at'):
                    update_expression += f", #{key} = :{key}"
                    expression_attribute_values[f":{key}"] = value
            
            # Build expression attribute names (to handle reserved words)
            expression_attribute_names = {
                f"#{key}": key for key in updates.keys() 
                if key not in ('id', 'created_at', 'updated_at')
            }
            
            if expression_attribute_names:
                table.update_item(
                    Key={'id': model_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_attribute_values,
                    ExpressionAttributeNames=expression_attribute_names
                )
            else:
                table.update_item(
                    Key={'id': model_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_attribute_values
                )
                
            return True
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {str(e)}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            bool: True if the deletion was successful
        """
        try:
            # First check if the model exists
            existing = self.get_model(model_id)
            if not existing:
                return False
                
            table = self.dynamodb.Table(f"{self.table_prefix}models")
            table.delete_item(Key={'id': model_id})
            
            # TODO: Also delete associated artifacts and metrics
            
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
            
    def save_model_artifact(self, model_id: str, version: str, artifact_data: Dict[str, Any]) -> bool:
        """
        Save a model artifact.
        
        Args:
            model_id (str): Unique identifier for the model
            version (str): Version of the artifact
            artifact_data (Dict[str, Any]): Artifact metadata and location
            
        Returns:
            bool: True if the operation was successful
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}model_artifacts")
            
            # Add timestamps
            artifact_data['created_at'] = int(time.time())
            
            # Add the keys to the item
            item = {
                'model_id': model_id,
                'version': version,
                **artifact_data
            }
            
            table.put_item(Item=item)
            return True
        except Exception as e:
            logger.error(f"Error saving artifact for model {model_id}: {str(e)}")
            return False
    
    def get_model_artifacts(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts for a model.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            List[Dict[str, Any]]: List of artifact data
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}model_artifacts")
            response = table.query(
                KeyConditionExpression="model_id = :model_id",
                ExpressionAttributeValues={
                    ':model_id': model_id
                }
            )
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error retrieving artifacts for model {model_id}: {str(e)}")
            return []
    
    def save_model_metric(self, model_id: str, metric_id: str, metric_data: Dict[str, Any]) -> bool:
        """
        Save a model metric.
        
        Args:
            model_id (str): Unique identifier for the model
            metric_id (str): Identifier for the metric
            metric_data (Dict[str, Any]): Metric data
            
        Returns:
            bool: True if the operation was successful
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}model_metrics")
            
            # Add timestamps
            metric_data['timestamp'] = int(time.time())
            
            # Add the keys to the item
            item = {
                'model_id': model_id,
                'metric_id': metric_id,
                **metric_data
            }
            
            table.put_item(Item=item)
            return True
        except Exception as e:
            logger.error(f"Error saving metric for model {model_id}: {str(e)}")
            return False
    
    def get_model_metrics(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a model.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            List[Dict[str, Any]]: List of metric data
        """
        try:
            table = self.dynamodb.Table(f"{self.table_prefix}model_metrics")
            response = table.query(
                KeyConditionExpression="model_id = :model_id",
                ExpressionAttributeValues={
                    ':model_id': model_id
                }
            )
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error retrieving metrics for model {model_id}: {str(e)}")
            return []