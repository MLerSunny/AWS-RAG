"""
A/B testing framework for comparing model performance.
"""
import random
from datetime import datetime
import uuid
import json
import os
from typing import Dict, List, Optional
import pandas as pd
from ..models.model_manager import ModelManager
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ABTestingFramework:
    """Framework for conducting A/B tests between different models."""
    
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
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Initialized A/B testing framework with storage in {storage_dir}")
        
    def create_experiment(self, 
                          experiment_id: str, 
                          model_ids: List[str], 
                          traffic_split: List[float] = None,
                          description: str = None):
        """
        Create a new A/B test experiment.
        
        Args:
            experiment_id (str): Unique identifier for the experiment
            model_ids (List[str]): List of model IDs to test
            traffic_split (List[float], optional): Percentage of traffic for each model
            description (str, optional): Description of the experiment
        """
        if not traffic_split:
            # Equal split by default
            traffic_split = [1.0/len(model_ids)] * len(model_ids)
            
        if sum(traffic_split) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")
            
        self.experiments[experiment_id] = {
            "model_ids": model_ids,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "description": description,
            "impressions": 0,
            "interactions": 0
        }
        
        # Save experiment configuration
        self._save_experiment(experiment_id)
        
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
                          timestamp: str = None):
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
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        impression_id = str(uuid.uuid4())
        timestamp = timestamp or datetime.now().isoformat()
        
        impression_data = {
            "impression_id": impression_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "model_id": model_id,
            "query": query,
            "response": response,
            "timestamp": timestamp
        }
        
        # Update impression count
        self.experiments[experiment_id]["impressions"] += 1
        
        # Save impression data
        self._save_impression(impression_data)
        
        logger.info(f"Recorded impression {impression_id} for experiment {experiment_id}")
        return impression_id
        
    def record_feedback(self, 
                        experiment_id: str, 
                        impression_id: str,
                        is_helpful: bool, 
                        feedback_text: str = "",
                        timestamp: str = None):
        """
        Record user feedback for a response.
        
        Args:
            experiment_id (str): Experiment identifier
            impression_id (str): Impression identifier
            is_helpful (bool): Whether the response was helpful
            feedback_text (str, optional): Additional feedback text
            timestamp (str, optional): Timestamp of the feedback
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        feedback_id = str(uuid.uuid4())
        timestamp = timestamp or datetime.now().isoformat()
        
        feedback_data = {
            "feedback_id": feedback_id,
            "experiment_id": experiment_id,
            "impression_id": impression_id,
            "is_helpful": is_helpful,
            "feedback_text": feedback_text,
            "timestamp": timestamp
        }
        
        # Update interaction count
        self.experiments[experiment_id]["interactions"] += 1
        
        # Save feedback data
        self._save_feedback(feedback_data)
        
        logger.info(f"Recorded feedback {feedback_id} for impression {impression_id}")
        return feedback_id
        
    def get_experiment_stats(self, experiment_id: str) -> Dict:
        """
        Get statistics for an experiment.
        
        Args:
            experiment_id (str): Experiment identifier
            
        Returns:
            Dict: Experiment statistics
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # Load impressions and feedback for this experiment
        impressions = self._load_impressions(experiment_id)
        feedback = self._load_feedback(experiment_id)
        
        # Process into a dataframe for analysis
        if not impressions:
            return {
                "status": "no_data",
                "models": self.experiments[experiment_id]["model_ids"],
                "impressions": 0,
                "interactions": 0
            }
        
        df_impressions = pd.DataFrame(impressions)
        
        # Create stats for each model
        models = {}
        for model_id in self.experiments[experiment_id]["model_ids"]:
            model_impressions = df_impressions[df_impressions["model_id"] == model_id]
            model_impression_ids = set(model_impressions["impression_id"])
            
            # Calculate feedback stats
            model_feedback = [f for f in feedback if f["impression_id"] in model_impression_ids]
            positive_feedback = sum(1 for f in model_feedback if f["is_helpful"])
            
            models[model_id] = {
                "impressions": len(model_impressions),
                "feedback_count": len(model_feedback),
                "positive_feedback": positive_feedback,
                "positive_rate": positive_feedback / len(model_feedback) if model_feedback else 0
            }
        
        return {
            "status": "active" if self.experiments[experiment_id]["active"] else "inactive",
            "created_at": self.experiments[experiment_id]["created_at"],
            "models": models,
            "total_impressions": len(impressions),
            "total_feedback": len(feedback)
        }
    
    def _save_experiment(self, experiment_id: str):
        """Save experiment configuration to file."""
        file_path = os.path.join(self.storage_dir, f"experiment_{experiment_id}.json")
        with open(file_path, 'w') as f:
            json.dump(self.experiments[experiment_id], f, indent=2)
    
    def _save_impression(self, impression_data: Dict):
        """Save impression data to file."""
        exp_id = impression_data["experiment_id"]
        file_path = os.path.join(self.storage_dir, f"impressions_{exp_id}.jsonl")
        with open(file_path, 'a') as f:
            f.write(json.dumps(impression_data) + "\n")
    
    def _save_feedback(self, feedback_data: Dict):
        """Save feedback data to file."""
        exp_id = feedback_data["experiment_id"]
        file_path = os.path.join(self.storage_dir, f"feedback_{exp_id}.jsonl")
        with open(file_path, 'a') as f:
            f.write(json.dumps(feedback_data) + "\n")
    
    def _load_impressions(self, experiment_id: str) -> List[Dict]:
        """Load impression data for an experiment."""
        file_path = os.path.join(self.storage_dir, f"impressions_{experiment_id}.jsonl")
        impressions = []
        
        if not os.path.exists(file_path):
            return impressions
            
        with open(file_path, 'r') as f:
            for line in f:
                impressions.append(json.loads(line.strip()))
                
        return impressions
    
    def _load_feedback(self, experiment_id: str) -> List[Dict]:
        """Load feedback data for an experiment."""
        file_path = os.path.join(self.storage_dir, f"feedback_{experiment_id}.jsonl")
        feedback = []
        
        if not os.path.exists(file_path):
            return feedback
            
        with open(file_path, 'r') as f:
            for line in f:
                feedback.append(json.loads(line.strip()))
                
        return feedback 