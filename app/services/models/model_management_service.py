from typing import Dict, List, Any, Optional, Mapping
from datetime import datetime
import json
import boto3
from app.utils.logger import setup_logger
from app.services.storage.dynamodb_service import DynamoDBService

logger = setup_logger(__name__)

class ModelManagementService:
    """Service for managing model versions, deployment, and performance."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        self.bedrock = boto3.client('bedrock')
        
    def register_model(self, model_data: Dict[str, Any]) -> str:
        """
        Register a new model version.
        
        Args:
            model_data: Model configuration and metadata
            
        Returns:
            Model ID
        """
        try:
            # Generate model ID
            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Add metadata
            model_data.update({
                "id": model_id,
                "created_at": datetime.utcnow().isoformat(),
                "status": "registered",
                "version": "1.0.0",
                "metrics": {}
            })
            
            # Save to DynamoDB
            table = self.db_service._get_table(f"{self.table_prefix}models")
            table.put_item(Item=model_data)
            
            logger.info(f"Registered new model: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def deploy_model(self, model_id: str, environment: str = "production") -> bool:
        """
        Deploy a model version to the specified environment.
        
        Args:
            model_id: ID of the model to deploy
            environment: Target environment
            
        Returns:
            True if deployment was successful
        """
        try:
            # Get model data
            table = self.db_service._get_table(f"{self.table_prefix}models")
            response = table.get_item(Key={"id": model_id})
            model_data = response.get("Item")
            
            if not model_data:
                raise ValueError(f"Model {model_id} not found")
            
            # Update model status
            model_data["status"] = "deploying"
            model_data["environment"] = environment
            model_data["deployed_at"] = datetime.utcnow().isoformat()
            
            # Ensure model_data is a dict before updating
            if not isinstance(model_data, dict):
                raise ValueError("Invalid model data format")
            
            table.put_item(Item=model_data)
            
            # Deploy to Bedrock
            if model_data.get("type") == "bedrock":
                self._deploy_bedrock_model(model_data)
            
            # Update status to deployed
            model_data["status"] = "deployed"
            table.put_item(Item=model_data)
            
            logger.info(f"Deployed model {model_id} to {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {str(e)}")
            
            # Update model status to failed
            try:
                if isinstance(model_data, dict):
                    model_data["status"] = "deployment_failed"
                    model_data["error"] = str(e)
                    table.put_item(Item=model_data)
            except:
                pass
            
            return False
    
    def _deploy_bedrock_model(self, model_data: Dict[str, Any]) -> None:
        """
        Deploy a model to AWS Bedrock.
        
        Args:
            model_data: Model configuration
        """
        try:
            # Create model in Bedrock
            self.bedrock.create_model(
                modelId=model_data["id"],
                modelName=model_data.get("name", model_data["id"]),
                modelArn=model_data["model_arn"],
                description=model_data.get("description", ""),
                tags=model_data.get("tags", {})
            )
        except Exception as e:
            logger.error(f"Error deploying to Bedrock: {str(e)}")
            raise
    
    def track_model_performance(self, model_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Track model performance metrics.
        
        Args:
            model_id: ID of the model
            metrics: Performance metrics
            
        Returns:
            True if tracking was successful
        """
        try:
            # Get current model data
            table = self.db_service._get_table(f"{self.table_prefix}models")
            response = table.get_item(Key={"id": model_id})
            model_data = response.get("Item")
            
            if not model_data:
                raise ValueError(f"Model {model_id} not found")
            
            # Ensure model_data is a dict
            if not isinstance(model_data, dict):
                raise ValueError("Invalid model data format")
            
            # Update metrics
            current_metrics = model_data.get("metrics", {})
            if isinstance(current_metrics, dict):
                current_metrics.update(metrics)
                current_metrics["last_updated"] = datetime.utcnow().isoformat()
                
                # Save updated metrics
                model_data["metrics"] = current_metrics
                table.put_item(Item=model_data)
                
                logger.info(f"Updated performance metrics for model {model_id}")
                return True
            else:
                raise ValueError("Invalid metrics format")
            
        except Exception as e:
            logger.error(f"Error tracking model performance: {str(e)}")
            return False
    
    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.
        
        Args:
            model_id: Base model ID
            
        Returns:
            List of model versions
        """
        try:
            table = self.db_service._get_table(f"{self.table_prefix}models")
            response = table.query(
                KeyConditionExpression="id = :id",
                ExpressionAttributeValues={":id": model_id}
            )
            return response.get("Items", [])
            
        except Exception as e:
            logger.error(f"Error getting model versions: {str(e)}")
            return []
    
    def rollback_model(self, model_id: str, version: str) -> bool:
        """
        Rollback a model to a previous version.
        
        Args:
            model_id: ID of the model
            version: Version to rollback to
            
        Returns:
            True if rollback was successful
        """
        try:
            # Get model versions
            versions = self.get_model_versions(model_id)
            if not versions:
                raise ValueError(f"No versions found for model {model_id}")
            
            # Find target version
            target_version = None
            for v in versions:
                if v["version"] == version:
                    target_version = v
                    break
            
            if not target_version:
                raise ValueError(f"Version {version} not found for model {model_id}")
            
            # Deploy target version
            return self.deploy_model(target_version["id"])
            
        except Exception as e:
            logger.error(f"Error rolling back model: {str(e)}")
            return False
    
    def get_model_metrics(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get model performance metrics over time.
        
        Args:
            model_id: ID of the model
            days: Number of days to look back
            
        Returns:
            Dict containing performance metrics
        """
        try:
            # Get model data
            table = self.db_service._get_table(f"{self.table_prefix}models")
            response = table.get_item(Key={"id": model_id})
            model_data = response.get("Item")
            
            if not model_data:
                raise ValueError(f"Model {model_id} not found")
            
            # Ensure model_data is a dict
            if not isinstance(model_data, dict):
                raise ValueError("Invalid model data format")
            
            # Get metrics history
            metrics = model_data.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            
            # Filter metrics by time range
            cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
            filtered_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) or k == "last_updated"
            }
            
            return {
                "model_id": model_id,
                "current_metrics": filtered_metrics,
                "history": metrics.get("history", [])
            }
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return {} 