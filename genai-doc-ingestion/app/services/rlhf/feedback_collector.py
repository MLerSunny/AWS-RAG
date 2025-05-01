"""
Feedback collection service for RLHF.
"""
import uuid
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class FeedbackEntry(BaseModel):
    """Model for a feedback entry."""
    response_id: str
    user_id: Optional[str] = None
    query: str
    model_id: str
    response_text: str
    is_helpful: bool
    feedback_text: Optional[str] = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = {}

class FeedbackCollector:
    """Service for collecting and processing user feedback for RLHF."""
    
    def __init__(self, storage_path: str = "data/feedback"):
        """
        Initialize the feedback collector.
        
        Args:
            storage_path (str, optional): Path to store feedback data
        """
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"Initialized feedback collector with storage in {storage_path}")
        
    def add_feedback(self, entry: FeedbackEntry):
        """
        Add a new feedback entry.
        
        Args:
            entry (FeedbackEntry): The feedback entry to add
        """
        # Save to storage
        self._save_feedback(entry.dict())
        logger.info(f"Added feedback for response {entry.response_id} from model {entry.model_id}")
        
    def get_feedback_for_model(self, model_id: str) -> List[Dict]:
        """
        Get all feedback for a specific model.
        
        Args:
            model_id (str): The model ID to filter by
            
        Returns:
            List[Dict]: List of feedback entries for the model
        """
        all_feedback = self._load_all_feedback()
        model_feedback = [entry for entry in all_feedback if entry["model_id"] == model_id]
        
        logger.info(f"Retrieved {len(model_feedback)} feedback entries for model {model_id}")
        return model_feedback
    
    def get_feedback_statistics(self) -> Dict:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dict: Feedback statistics
        """
        all_feedback = self._load_all_feedback()
        
        if not all_feedback:
            return {"total": 0, "models": {}}
            
        # Group by model
        models = {}
        for entry in all_feedback:
            model_id = entry["model_id"]
            if model_id not in models:
                models[model_id] = {
                    "total": 0,
                    "helpful": 0,
                    "not_helpful": 0
                }
                
            models[model_id]["total"] += 1
            if entry["is_helpful"]:
                models[model_id]["helpful"] += 1
            else:
                models[model_id]["not_helpful"] += 1
        
        # Calculate percentages
        for model_id, stats in models.items():
            stats["helpful_percent"] = (stats["helpful"] / stats["total"]) * 100
            stats["not_helpful_percent"] = (stats["not_helpful"] / stats["total"]) * 100
            
        return {
            "total": len(all_feedback),
            "models": models
        }
        
    def export_training_data(self, output_file: str, format_type: str = "preference_pairs"):
        """
        Export feedback as training data for RLHF.
        
        Args:
            output_file (str): Path to save the training data
            format_type (str, optional): Format of the training data
                Options: preference_pairs, binary_classification
        """
        all_feedback = self._load_all_feedback()
        
        if not all_feedback:
            logger.warning("No feedback data available for export")
            return
            
        if format_type == "preference_pairs":
            # Group feedback by query to create preference pairs
            query_groups = {}
            for entry in all_feedback:
                query = entry["query"]
                if query not in query_groups:
                    query_groups[query] = []
                query_groups[query].append(entry)
                
            # Create preference pairs
            pairs = []
            for query, entries in query_groups.items():
                if len(entries) < 2:
                    continue
                    
                # Sort by helpfulness, most helpful first
                sorted_entries = sorted(entries, key=lambda x: (x["is_helpful"], x["feedback_text"] != ""), reverse=True)
                
                # Take the most and least helpful responses
                chosen = sorted_entries[0]
                rejected = sorted_entries[-1]
                
                # Only create pairs where there's a clear preference
                if chosen["is_helpful"] and not rejected["is_helpful"]:
                    pairs.append({
                        "prompt": query,
                        "chosen": chosen["response_text"],
                        "rejected": rejected["response_text"]
                    })
                    
            # Write to file
            with open(output_file, 'w') as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                    
            logger.info(f"Exported {len(pairs)} preference pairs to {output_file}")
                
        elif format_type == "binary_classification":
            # Create binary classification examples
            examples = []
            for entry in all_feedback:
                examples.append({
                    "query": entry["query"],
                    "response": entry["response_text"],
                    "label": 1 if entry["is_helpful"] else 0
                })
                
            # Write to file
            with open(output_file, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")
                    
            logger.info(f"Exported {len(examples)} classification examples to {output_file}")
            
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _save_feedback(self, feedback_data: Dict):
        """Save feedback data to storage."""
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"feedback_{timestamp}_{unique_id}.json"
        
        file_path = os.path.join(self.storage_path, filename)
        
        with open(file_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
            
    def _load_all_feedback(self) -> List[Dict]:
        """Load all feedback data from storage."""
        feedback_files = [f for f in os.listdir(self.storage_path) if f.startswith("feedback_") and f.endswith(".json")]
        
        all_feedback = []
        for filename in feedback_files:
            file_path = os.path.join(self.storage_path, filename)
            try:
                with open(file_path, 'r') as f:
                    feedback_data = json.load(f)
                    all_feedback.append(feedback_data)
            except Exception as e:
                logger.error(f"Error loading feedback from {file_path}: {str(e)}")
                
        return all_feedback 