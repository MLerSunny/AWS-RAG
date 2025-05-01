"""
A/B testing framework for comparing model performance.
This module is deprecated. Please use app.services.experiment.ab_testing instead.
"""
import warnings
from typing import Dict, List, Optional, Union, Any
from ..models.model_manager import ModelManager
from ..experiment.ab_testing import ABTestingManager, AllocationStrategy
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

# Issue deprecation warning
warnings.warn(
    "The ABTestingFramework in app.services.evaluation.ab_testing is deprecated. "
    "Please use ABTestingManager from app.services.experiment.ab_testing instead.",
    DeprecationWarning,
    stacklevel=2
)

class ABTestingFramework:
    """
    Framework for conducting A/B tests between different models.
    DEPRECATED: Use ABTestingManager from app.services.experiment.ab_testing instead.
    """
    
    def __init__(self, model_manager: ModelManager, storage_dir: str = "data/ab_testing"):
        """
        Initialize the A/B testing framework.
        
        Args:
            model_manager (ModelManager): Model manager instance
            storage_dir (str, optional): Directory to store testing data
        """
        self.model_manager = model_manager
        self.storage_dir = storage_dir
        self.experiments = {}
        
        # Use the new ABTestingManager internally
        self._manager = ABTestingManager(storage_path=storage_dir)
        
        logger.info(f"Initialized A/B testing framework with storage in {storage_dir}")
        logger.warning("This class is deprecated. Please use ABTestingManager instead.")
        
    def create_experiment(self, 
                          experiment_id: str, 
                          model_ids: List[str], 
                          traffic_split: Optional[List[float]] = None,
                          description: Optional[str] = None):
        """
        Create a new A/B test experiment.
        
        Args:
            experiment_id (str): Unique identifier for the experiment
            model_ids (List[str]): List of model IDs to test
            traffic_split (List[float], optional): Percentage of traffic for each model
            description (str, optional): Description of the experiment
        """
        # Create variants for the new manager
        variants = []
        for i, model_id in enumerate(model_ids):
            weight = traffic_split[i] if traffic_split else 1.0
            variants.append({
                "name": f"Variant {i+1}",
                "model_id": model_id,
                "weight": weight
            })
        
        # Create the experiment with the new manager
        experiment = self._manager.create_experiment(
            name=experiment_id,
            variants=variants,
            description=description,
            allocation_strategy=AllocationStrategy.CUSTOM if traffic_split else AllocationStrategy.EQUAL
        )
        
        # Save for backward compatibility
        self.experiments[experiment_id] = {
            "model_ids": model_ids,
            "traffic_split": traffic_split or [1.0/len(model_ids)] * len(model_ids),
            "created_at": experiment.created_at,
            "active": True,
            "description": description,
            "impressions": 0,
            "interactions": 0
        }
        
        logger.info(f"Created experiment {experiment_id} with models: {', '.join(model_ids)}")
        
    def select_model_for_user(self, experiment_id: str, user_id: str) -> str:
        """
        Select a model for a user based on the experiment configuration.
        
        Args:
            experiment_id (str): Experiment identifier
            user_id (str): User identifier
            
        Returns:
            str: Selected model ID
        """
        # Use the new manager to select a variant
        variant = self._manager.select_variant(experiment_id, user_id)
        
        if variant:
            return variant.get("model_id", "")
        
        # Fallback to old method if new manager fails
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        
        # Use a hash of the user_id to ensure consistent model assignment
        hash_val = hash(user_id) % 100
        cumulative = 0
        
        for i, split in enumerate(experiment["traffic_split"]):
            cumulative += split * 100
            if hash_val < cumulative:
                return experiment["model_ids"][i]
                
        # Fallback to the last model
        return experiment["model_ids"][-1]
        
    def record_impression(self, 
                          experiment_id: str, 
                          user_id: str, 
                          model_id: str, 
                          query: str,
                          response: str,
                          timestamp: Optional[str] = None):
        """
        Record an impression in the experiment.
        
        Args:
            experiment_id (str): Experiment identifier
            user_id (str): User identifier
            model_id (str): Model that generated the response
            query (str): User query
            response (str): Model response
            timestamp (str, optional): Timestamp of the impression
        """
        # Update impression count in old format
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["impressions"] += 1
        
        # Find variant ID from model ID
        experiment = self._manager.get_experiment(experiment_id)
        if experiment:
            for variant in experiment.variants:
                if variant.model_id == model_id:
                    # Log impression in new manager
                    experiment.log_impression(variant.id)
                    break
        
        logger.info(f"Recorded impression for experiment {experiment_id}")
        return "impression-" + user_id
        
    def record_feedback(self, 
                        experiment_id: str, 
                        impression_id: str,
                        is_helpful: bool, 
                        feedback_text: str = "",
                        timestamp: Optional[str] = None):
        """
        Record user feedback for a response.
        
        Args:
            experiment_id (str): Experiment identifier
            impression_id (str): Impression identifier
            is_helpful (bool): Whether the response was helpful
            feedback_text (str, optional): Additional feedback text
            timestamp (str, optional): Timestamp of the feedback
        """
        # Extract user_id from impression_id (simplified)
        user_id = impression_id.replace("impression-", "")
        
        # Update interaction count in old format
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["interactions"] += 1
        
        # Record feedback using the new manager
        variant = self._manager.select_variant(experiment_id, user_id)
        if variant:
            variant_id = variant.get("id", "")
            self._manager.record_feedback(experiment_id, variant_id, is_helpful)
        
        logger.info(f"Recorded feedback for impression {impression_id}")
        return "feedback-" + user_id
        
    def get_experiment_stats(self, experiment_id: str) -> Dict:
        """
        Get statistics for an experiment.
        
        Args:
            experiment_id (str): Experiment identifier
            
        Returns:
            Dict: Experiment statistics
        """
        # Use the new manager to get stats
        stats = self._manager.get_experiment_stats(experiment_id)
        
        if "error" not in stats:
            return stats
            
        # Fallback to simplified stats if new manager fails
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        
        return {
            "experiment_id": experiment_id,
            "description": experiment.get("description", ""),
            "model_ids": experiment.get("model_ids", []),
            "impressions": experiment.get("impressions", 0),
            "interactions": experiment.get("interactions", 0),
            "active": experiment.get("active", False)
        } 