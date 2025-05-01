"""
Model management service for multiple LLM providers.
"""
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional, List
import json
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelType(str, Enum):
    """Enum for supported model types."""
    RAG = "rag"
    BEDROCK_BASE = "bedrock_base"
    BEDROCK_FINETUNED = "bedrock_finetuned"
    TITAN = "titan"
    DEEPSEEK = "deepseek"
    LLAMA = "llama"

class ModelConfig(BaseModel):
    """Configuration for a model."""
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    parameters: Dict = {}
    description: Optional[str] = None

class ModelManager:
    """Manager for multiple model types and configurations."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        logger.info("Initialized ModelManager")
        
    def register_model(self, model_id: str, config: ModelConfig):
        """
        Register a model configuration.
        
        Args:
            model_id (str): Unique identifier for the model
            config (ModelConfig): Configuration for the model
        """
        self.models[model_id] = config
        logger.info(f"Registered model {model_id} of type {config.type}")
        
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a model configuration by ID.
        
        Args:
            model_id (str): The model ID
            
        Returns:
            Optional[ModelConfig]: The model configuration or None if not found
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found")
            return None
            
        return self.models[model_id]
        
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns:
            List[Dict]: List of model information
        """
        return [
            {
                "id": model_id,
                "name": config.name,
                "type": config.type,
                "description": config.description
            }
            for model_id, config in self.models.items()
        ]
        
    def list_models_by_type(self, model_type: ModelType) -> List[Dict]:
        """
        List models of a specific type.
        
        Args:
            model_type (ModelType): The model type to filter by
            
        Returns:
            List[Dict]: List of models matching the type
        """
        return [
            {
                "id": model_id,
                "name": config.name,
                "type": config.type,
                "description": config.description
            }
            for model_id, config in self.models.items()
            if config.type == model_type
        ]
    
    def save_to_file(self, file_path: str):
        """
        Save the model registry to a file.
        
        Args:
            file_path (str): Path to save the registry
        """
        try:
            model_data = {
                model_id: config.dict() 
                for model_id, config in self.models.items()
            }
            
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            logger.info(f"Saved model registry to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
            raise
    
    def load_from_file(self, file_path: str):
        """
        Load the model registry from a file.
        
        Args:
            file_path (str): Path to load the registry from
        """
        try:
            with open(file_path, 'r') as f:
                model_data = json.load(f)
                
            for model_id, config_dict in model_data.items():
                self.register_model(model_id, ModelConfig(**config_dict))
                
            logger.info(f"Loaded model registry from {file_path} with {len(model_data)} models")
        except Exception as e:
            logger.error(f"Error loading model registry: {str(e)}")
            raise 