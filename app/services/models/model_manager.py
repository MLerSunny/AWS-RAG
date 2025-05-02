"""
Model management service for multiple LLM providers.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Set, Union, Any
import json
import os
import time
import functools
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
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelManager:
    """Manager for multiple model types and configurations."""
    
    def __init__(self, auto_save_path: Optional[str] = None, load_on_start: bool = False):
        """
        Initialize the model manager.
        
        Args:
            auto_save_path (Optional[str]): Path to automatically save model registry
            load_on_start (bool): Whether to load the registry on startup
        """
        self.models = {}
        self.auto_save_path = auto_save_path
        self._last_list_time = 0
        self._cached_model_list = []
        
        # Create directory for auto-save if needed
        if auto_save_path:
            os.makedirs(os.path.dirname(auto_save_path), exist_ok=True)
            
            # Load existing models if requested
            if load_on_start and os.path.exists(auto_save_path):
                try:
                    self.load_from_file(auto_save_path)
                except Exception as e:
                    logger.error(f"Error loading model registry: {str(e)}")
                
        logger.info("Initialized ModelManager")
    
    def register_model(self, model_id: str, config: ModelConfig):
        """
        Register a model configuration.
        
        Args:
            model_id (str): Unique identifier for the model
            config (ModelConfig): Configuration for the model
        """
        # Set timestamps if not already set
        if not config.created_at:
            config.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        config.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # Add to registry
        self.models[model_id] = config
        logger.info(f"Registered model {model_id} of type {config.type}")
        
        # Auto-save if path is specified
        if self.auto_save_path:
            try:
                self.save_to_file(self.auto_save_path)
            except Exception as e:
                logger.error(f"Error auto-saving model registry: {str(e)}")
        
        # Invalidate cache
        self._last_list_time = 0
    
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
        # Check if we can use cached result
        current_time = time.time()
        if current_time - self._last_list_time < 60 and self._cached_model_list:  # 60 second cache
            return self._cached_model_list
        
        # Generate fresh list
        result = [
            {
                "id": model_id,
                "name": config.name,
                "type": config.type,
                "description": config.description,
                "enabled": config.enabled
            }
            for model_id, config in self.models.items()
            if config.enabled  # Only show enabled models
        ]
        
        # Update cache
        self._cached_model_list = result
        self._last_list_time = current_time
        
        return result
    
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
                "description": config.description,
                "enabled": config.enabled
            }
            for model_id, config in self.models.items()
            if config.type == model_type and config.enabled
        ]
    
    def update_model(self, model_id: str, config_updates: Dict[str, Any]) -> bool:
        """
        Update a model configuration.
        
        Args:
            model_id (str): The model ID to update
            config_updates (Dict[str, Any]): Updates to apply to the model
            
        Returns:
            bool: True if update was successful
        """
        if model_id not in self.models:
            logger.warning(f"Cannot update model {model_id} - not found")
            return False
        
        # Get existing config
        existing_config = self.models[model_id]
        
        # Update fields
        for key, value in config_updates.items():
            if hasattr(existing_config, key):
                setattr(existing_config, key, value)
        
        # Update timestamp
        existing_config.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # Auto-save if path is specified
        if self.auto_save_path:
            try:
                self.save_to_file(self.auto_save_path)
            except Exception as e:
                logger.error(f"Error auto-saving model registry: {str(e)}")
        
        # Invalidate cache
        self._last_list_time = 0
        
        logger.info(f"Updated model {model_id}")
        return True
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id (str): The model ID to delete
            
        Returns:
            bool: True if deletion was successful
        """
        if model_id not in self.models:
            logger.warning(f"Cannot delete model {model_id} - not found")
            return False
        
        # Remove from registry
        del self.models[model_id]
        
        # Auto-save if path is specified
        if self.auto_save_path:
            try:
                self.save_to_file(self.auto_save_path)
            except Exception as e:
                logger.error(f"Error auto-saving model registry: {str(e)}")
        
        # Invalidate cache
        self._last_list_time = 0
        
        logger.info(f"Deleted model {model_id}")
        return True
    
    def save_to_file(self, file_path: str):
        """
        Save the model registry to a file.
        
        Args:
            file_path (str): Path to save the registry
        """
        try:
            # Convert models to dict format
            model_data = {
                model_id: config.dict() 
                for model_id, config in self.models.items()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
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
            if not os.path.exists(file_path):
                logger.warning(f"Model registry file not found: {file_path}")
                return
                
            with open(file_path, 'r') as f:
                model_data = json.load(f)
                
            # Clear existing models
            self.models = {}
            
            # Register models from file
            for model_id, config_dict in model_data.items():
                try:
                    self.register_model(model_id, ModelConfig(**config_dict))
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {str(e)}")
                
            logger.info(f"Loaded model registry from {file_path} with {len(model_data)} models")
            
            # Invalidate cache
            self._last_list_time = 0
        except Exception as e:
            logger.error(f"Error loading model registry: {str(e)}")
            raise 