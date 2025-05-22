"""
DynamoDB service for persistent storage of application state.
"""
import boto3
import time
import os
import json
import re
from typing import Dict, List, Optional, Any, Union, cast, TypeVar, Generic, Iterator, Tuple, Sequence
from botocore.exceptions import ClientError, WaiterError
from boto3.resources.base import ServiceResource
from boto3.dynamodb.conditions import Key
from mypy_boto3_dynamodb import DynamoDBServiceResource
from mypy_boto3_dynamodb.service_resource import Table
from mypy_boto3_dynamodb.type_defs import KeySchemaElementTypeDef, AttributeDefinitionTypeDef
from app.utils.logger import setup_logger
from app.utils.validation import validate_not_empty, validate_dict, validate_type, validate_range

logger = setup_logger(__name__)

class DynamoDBError(Exception):
    """Base exception for DynamoDB service errors."""
    def __init__(self, message: str, error_code: str = "DYNAMODB_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class TableNotFoundError(DynamoDBError):
    """Raised when a table is not found."""
    def __init__(self, message: str):
        super().__init__(message, "TABLE_NOT_FOUND")

class BatchOperationError(DynamoDBError):
    """Raised when a batch operation fails."""
    def __init__(self, message: str, failed_items: List[Dict[str, Any]]):
        super().__init__(message, "BATCH_OPERATION_FAILED")
        self.failed_items = failed_items

class TableCreationError(DynamoDBError):
    """Raised when table creation fails."""
    def __init__(self, message: str):
        super().__init__(message, "TABLE_CREATION_FAILED")

T = TypeVar('T')

class DynamoDBService:
    """Service for interacting with DynamoDB tables."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    BATCH_SIZE = 25  # DynamoDB batch operation limit
    MAX_LIMIT = 1000  # Maximum number of items to return in list operations
    TABLE_NAME_PATTERN = r'^[a-zA-Z0-9_.-]{3,255}$'  # DynamoDB table name pattern
    
    def __init__(self, table_prefix: str = "genai_"):
        """
        Initialize the DynamoDB service.
        
        Args:
            table_prefix (str): Prefix for DynamoDB table names
        """
        validate_not_empty(table_prefix, "table_prefix")
        if not re.match(self.TABLE_NAME_PATTERN, table_prefix):
            raise ValueError(f"Invalid table prefix: {table_prefix}. Must match pattern: {self.TABLE_NAME_PATTERN}")
        
        # Use local endpoint for local development if specified
        endpoint_url = os.environ.get("DYNAMODB_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using local DynamoDB endpoint: {endpoint_url}")
            self.dynamodb: DynamoDBServiceResource = cast(DynamoDBServiceResource, boto3.resource('dynamodb', endpoint_url=endpoint_url))
        else:
            self.dynamodb: DynamoDBServiceResource = cast(DynamoDBServiceResource, boto3.resource('dynamodb'))
            
        self.table_prefix = table_prefix
        self._ensure_tables()
    
    def _validate_table_name(self, table_name: str) -> None:
        """
        Validate a table name.
        
        Args:
            table_name (str): Name of the table to validate
            
        Raises:
            ValueError: If the table name is invalid
        """
        validate_not_empty(table_name, "table_name")
        if not re.match(self.TABLE_NAME_PATTERN, table_name):
            raise ValueError(f"Invalid table name: {table_name}. Must match pattern: {self.TABLE_NAME_PATTERN}")
    
    def _get_table(self, table_name: str) -> Table:
        """
        Get a DynamoDB table with retry logic.
        
        Args:
            table_name (str): Name of the table to get
            
        Returns:
            Table: DynamoDB table resource
            
        Raises:
            TableNotFoundError: If table cannot be accessed after retries
        """
        self._validate_table_name(table_name)
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.dynamodb.Table(table_name)
            except ClientError as e:
                last_error = e
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Failed to get table {table_name} after {self.MAX_RETRIES} attempts: {str(e)}")
                    raise TableNotFoundError(f"Table {table_name} not found: {str(e)}")
                logger.warning(f"Failed to get table {table_name}, attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}")
                time.sleep(self.RETRY_DELAY)
        
        # This should never be reached due to the raise in the loop, but needed for type checker
        raise last_error or TableNotFoundError(f"Table {table_name} not found")
    
    def _wait_for_table(self, table_name: str) -> None:
        """
        Wait for a table to become active.
        
        Args:
            table_name (str): Name of the table to wait for
            
        Raises:
            WaiterError: If table does not become active within timeout
        """
        self._validate_table_name(table_name)
        
        try:
            waiter = self.dynamodb.meta.client.get_waiter('table_exists')
            waiter.wait(TableName=table_name)
        except WaiterError as e:
            logger.error(f"Failed to wait for table {table_name}: {str(e)}")
            raise TableCreationError(f"Failed to wait for table {table_name}: {str(e)}")
    
    def _create_table(self, table_name: str, key_schema: Sequence[KeySchemaElementTypeDef], attribute_definitions: Sequence[AttributeDefinitionTypeDef]) -> None:
        """
        Create a DynamoDB table.
        
        Args:
            table_name (str): Name of the table to create
            key_schema (Sequence[KeySchemaElementTypeDef]): Key schema for the table
            attribute_definitions (Sequence[AttributeDefinitionTypeDef]): Attribute definitions for the table
            
        Raises:
            TableCreationError: If table creation fails
        """
        self._validate_table_name(table_name)
        
        try:
            logger.info(f"Creating table: {table_name}")
            self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attribute_definitions,
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            self._wait_for_table(table_name)
            logger.info(f"Created table: {table_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.info(f"Table {table_name} already exists")
            else:
                logger.error(f"Error creating table {table_name}: {e}")
                raise TableCreationError(f"Failed to create table {table_name}: {str(e)}")
    
    def _batch_delete_items(self, table: Table, items: List[Dict[str, Any]], key_attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Delete items in batches.
        
        Args:
            table: DynamoDB table
            items: List of items to delete
            key_attributes: List of key attribute names
            
        Returns:
            List[Dict[str, Any]]: List of failed items
            
        Raises:
            BatchOperationError: If batch operation fails
        """
        failed_items: List[Dict[str, Any]] = []
        
        try:
            with table.batch_writer() as batch:
                for item in items:
                    try:
                        key = {attr: item[attr] for attr in key_attributes}
                        batch.delete_item(Key=key)
                    except Exception as e:
                        logger.error(f"Failed to delete item {item}: {str(e)}")
                        failed_items.append(item)
        except Exception as e:
            logger.error(f"Batch delete operation failed: {str(e)}")
            raise BatchOperationError(f"Batch delete operation failed: {str(e)}", failed_items)
        
        return failed_items
    
    def _ensure_tables(self) -> None:
        """Ensure required DynamoDB tables exist."""
        tables = list(self.dynamodb.tables.all())
        table_names = [table.name for table in tables]
        
        # Model registry table
        model_table_name = f"{self.table_prefix}models"
        if model_table_name not in table_names:
            self._create_table(
                model_table_name,
                [
                    {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
                ],
                [
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ]
            )
        
        # Model artifacts table
        artifacts_table_name = f"{self.table_prefix}model_artifacts"
        if artifacts_table_name not in table_names:
            self._create_table(
                artifacts_table_name,
                [
                    {'AttributeName': 'model_id', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'version', 'KeyType': 'RANGE'}   # Sort key
                ],
                [
                    {'AttributeName': 'model_id', 'AttributeType': 'S'},
                    {'AttributeName': 'version', 'AttributeType': 'S'}
                ]
            )
        
        # Model metrics table
        metrics_table_name = f"{self.table_prefix}model_metrics"
        if metrics_table_name not in table_names:
            self._create_table(
                metrics_table_name,
                [
                    {'AttributeName': 'model_id', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'metric_id', 'KeyType': 'RANGE'}  # Sort key
                ],
                [
                    {'AttributeName': 'model_id', 'AttributeType': 'S'},
                    {'AttributeName': 'metric_id', 'AttributeType': 'S'}
                ]
            )
        
        # User interactions table
        self.ensure_user_interactions_table()

    def ensure_user_interactions_table(self):
        """
        Ensure the 'genai_user_interactions' table exists with 'pk' as the primary key.
        """
        table_name = f"{self.table_prefix}user_interactions" if not self.table_prefix.endswith('_') else f"{self.table_prefix}user_interactions"
        try:
            self._get_table(table_name)
        except TableNotFoundError:
            key_schema: Sequence[KeySchemaElementTypeDef] = [
                {"AttributeName": "pk", "KeyType": "HASH"}  # type: ignore
            ]
            attribute_definitions: Sequence[AttributeDefinitionTypeDef] = [
                {"AttributeName": "pk", "AttributeType": "S"}  # type: ignore
            ]
            self._create_table(table_name, key_schema, attribute_definitions)

    def save_model(self, model_id: str, model_data: Dict[str, Any]) -> bool:
        """
        Save a model to the model registry.
        
        Args:
            model_id (str): Unique identifier for the model
            model_data (Dict[str, Any]): Model configuration data
            
        Returns:
            bool: True if the operation was successful
        """
        validate_not_empty(model_id, "model_id")
        validate_dict(model_data, "model_data")
        
        try:
            table = self._get_table(f"{self.table_prefix}models")
            
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
        validate_not_empty(model_id, "model_id")
        
        try:
            table = self._get_table(f"{self.table_prefix}models")
            response = table.get_item(Key={'id': model_id})
            
            if 'Item' in response:
                return response['Item']
            return None
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {str(e)}")
            return None
    
    def list_models(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Args:
            limit (Optional[int]): Maximum number of models to return
            
        Returns:
            List[Dict[str, Any]]: List of model configurations
        """
        if limit is not None:
            validate_type(limit, int, "limit")
            validate_range(limit, "limit", min_value=1, max_value=self.MAX_LIMIT)
        
        try:
            table = self._get_table(f"{self.table_prefix}models")
            items: List[Dict[str, Any]] = []
            
            # Use pagination to handle large datasets
            last_evaluated_key = None
            while True:
                if limit and len(items) >= limit:
                    break
                    
                scan_kwargs = {}
                if last_evaluated_key:
                    scan_kwargs['ExclusiveStartKey'] = last_evaluated_key
                
                response = table.scan(**scan_kwargs)
                items.extend(response.get('Items', []))
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return items[:limit] if limit else items
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
        validate_not_empty(model_id, "model_id")
        validate_dict(updates, "updates")
        
        try:
            table = self._get_table(f"{self.table_prefix}models")
            
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
        Delete a model from the registry and all associated resources.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            bool: True if the deletion was successful
        """
        validate_not_empty(model_id, "model_id")
        
        try:
            # First check if the model exists
            existing = self.get_model(model_id)
            if not existing:
                return False
            
            # Delete the model
            table = self._get_table(f"{self.table_prefix}models")
            table.delete_item(Key={'id': model_id})
            
            # Delete associated artifacts
            artifacts = self.get_model_artifacts(model_id)
            if artifacts:
                artifacts_table = self._get_table(f"{self.table_prefix}model_artifacts")
                failed_artifacts = self._batch_delete_items(
                    artifacts_table,
                    artifacts,
                    ['model_id', 'version']
                )
                if failed_artifacts:
                    logger.warning(f"Failed to delete {len(failed_artifacts)} artifacts for model {model_id}")
            
            # Delete associated metrics
            metrics = self.get_model_metrics(model_id)
            if metrics:
                metrics_table = self._get_table(f"{self.table_prefix}model_metrics")
                failed_metrics = self._batch_delete_items(
                    metrics_table,
                    metrics,
                    ['model_id', 'metric_id']
                )
                if failed_metrics:
                    logger.warning(f"Failed to delete {len(failed_metrics)} metrics for model {model_id}")
            
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
        validate_not_empty(model_id, "model_id")
        validate_not_empty(version, "version")
        validate_dict(artifact_data, "artifact_data")
        
        try:
            table = self._get_table(f"{self.table_prefix}model_artifacts")
            
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
    
    def get_model_artifacts(self, model_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all artifacts for a model.
        
        Args:
            model_id (str): Unique identifier for the model
            limit (Optional[int]): Maximum number of artifacts to return
            
        Returns:
            List[Dict[str, Any]]: List of artifact data
        """
        validate_not_empty(model_id, "model_id")
        if limit is not None:
            validate_type(limit, int, "limit")
            validate_range(limit, "limit", min_value=1, max_value=self.MAX_LIMIT)
        
        try:
            table = self._get_table(f"{self.table_prefix}model_artifacts")
            items: List[Dict[str, Any]] = []
            
            # Use pagination to handle large datasets
            last_evaluated_key = None
            while True:
                if limit and len(items) >= limit:
                    break
                    
                query_kwargs = {
                    'KeyConditionExpression': "model_id = :model_id",
                    'ExpressionAttributeValues': {
                        ':model_id': model_id
                    }
                }
                if last_evaluated_key:
                    query_kwargs['ExclusiveStartKey'] = last_evaluated_key
                
                response = table.query(**query_kwargs)
                items.extend(response.get('Items', []))
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return items[:limit] if limit else items
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
        validate_not_empty(model_id, "model_id")
        validate_not_empty(metric_id, "metric_id")
        validate_dict(metric_data, "metric_data")
        
        try:
            table = self._get_table(f"{self.table_prefix}model_metrics")
            
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
    
    def get_model_metrics(self, model_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all metrics for a model.
        
        Args:
            model_id (str): Unique identifier for the model
            limit (Optional[int]): Maximum number of metrics to return
            
        Returns:
            List[Dict[str, Any]]: List of metric data
        """
        validate_not_empty(model_id, "model_id")
        if limit is not None:
            validate_type(limit, int, "limit")
            validate_range(limit, "limit", min_value=1, max_value=self.MAX_LIMIT)
        
        try:
            table = self._get_table(f"{self.table_prefix}model_metrics")
            items: List[Dict[str, Any]] = []
            
            # Use pagination to handle large datasets
            last_evaluated_key = None
            while True:
                if limit and len(items) >= limit:
                    break
                    
                query_kwargs = {
                    'KeyConditionExpression': "model_id = :model_id",
                    'ExpressionAttributeValues': {
                        ':model_id': model_id
                    }
                }
                if last_evaluated_key:
                    query_kwargs['ExclusiveStartKey'] = last_evaluated_key
                
                response = table.query(**query_kwargs)
                items.extend(response.get('Items', []))
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return items[:limit] if limit else items
        except Exception as e:
            logger.error(f"Error retrieving metrics for model {model_id}: {str(e)}")
            return []