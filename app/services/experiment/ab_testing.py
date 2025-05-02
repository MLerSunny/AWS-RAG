"""
A/B Testing framework for comparing model performance.
"""
from typing import Dict, List, Optional, Any, Union
import json
import time
import random
import uuid
import os
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class AllocationStrategy(str, Enum):
    """Strategies for allocating users to variants."""
    EQUAL = "equal"           # Equal distribution across variants
    THOMPSON = "thompson"     # Thompson sampling based on feedback
    CUSTOM = "custom"         # Custom weights

class Variant(BaseModel):
    """Definition of an experiment variant."""
    id: str
    name: str
    model_id: str
    description: Optional[str] = None
    weight: float = 1.0       # Relative weight for allocation
    
    # Stats
    impressions: int = 0      # Number of times variant was shown
    positive_feedback: int = 0
    negative_feedback: int = 0
    
    @property
    def total_feedback(self) -> int:
        """Total feedback count."""
        return self.positive_feedback + self.negative_feedback
    
    @property
    def success_rate(self) -> float:
        """Success rate based on feedback."""
        if self.total_feedback == 0:
            return 0.5  # Default for no data
        return self.positive_feedback / self.total_feedback
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "model_id": self.model_id,
            "description": self.description,
            "weight": self.weight,
            "stats": {
                "impressions": self.impressions,
                "positive_feedback": self.positive_feedback,
                "negative_feedback": self.negative_feedback,
                "total_feedback": self.total_feedback,
                "success_rate": self.success_rate
            }
        }

class Experiment(BaseModel):
    """Definition of an A/B testing experiment."""
    id: str
    name: str
    description: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant]
    allocation_strategy: AllocationStrategy = AllocationStrategy.EQUAL
    
    # Time bounds
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Targeting
    user_segment: Optional[Dict[str, Any]] = None
    
    def add_variant(self, variant: Variant) -> None:
        """Add a variant to the experiment."""
        self.variants.append(variant)
    
    def update_variant(self, variant_id: str, updates: Dict[str, Any]) -> bool:
        """Update a variant in the experiment."""
        for i, variant in enumerate(self.variants):
            if variant.id == variant_id:
                for key, value in updates.items():
                    if hasattr(variant, key):
                        setattr(variant, key, value)
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "allocation_strategy": self.allocation_strategy,
            "created_at": self.created_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "user_segment": self.user_segment,
            "variants": [v.to_dict() for v in self.variants],
        }
    
    def is_active(self) -> bool:
        """Check if the experiment is currently active."""
        if self.status != ExperimentStatus.ACTIVE:
            return False
            
        now = datetime.now()
        
        # Check start date
        if self.start_date:
            start_date = datetime.fromisoformat(self.start_date)
            if now < start_date:
                return False
                
        # Check end date
        if self.end_date:
            end_date = datetime.fromisoformat(self.end_date)
            if now > end_date:
                return False
                
        return True
    
    def get_variant_for_user(self, user_id: str) -> Optional[Variant]:
        """
        Select a variant for a specific user.
        
        Args:
            user_id (str): The user identifier
            
        Returns:
            Optional[Variant]: The selected variant or None
        """
        if not self.is_active() or not self.variants:
            return None
            
        if self.allocation_strategy == AllocationStrategy.EQUAL:
            # Simple hash-based consistent assignment
            # Ensures the same user always gets the same variant
            variant_index = hash(user_id) % len(self.variants)
            return self.variants[variant_index]
            
        elif self.allocation_strategy == AllocationStrategy.THOMPSON:
            # Thompson sampling for multi-armed bandit
            # Randomly select based on success rate
            samples = []
            for variant in self.variants:
                # Beta distribution based on successes and failures
                alpha = variant.positive_feedback + 1  # +1 to avoid zero
                beta = variant.negative_feedback + 1   # +1 to avoid zero
                sample = random.betavariate(alpha, beta)
                samples.append((sample, variant))
            
            # Select variant with highest sample value
            selected_variant = max(samples, key=lambda x: x[0])[1]
            return selected_variant
            
        elif self.allocation_strategy == AllocationStrategy.CUSTOM:
            # Use defined weights
            weights = [v.weight for v in self.variants]
            total_weight = sum(weights)
            
            # Normalize weights
            normalized_weights = [w / total_weight for w in weights]
            
            # Select based on weights
            r = random.random()
            cumulative = 0
            for i, weight in enumerate(normalized_weights):
                cumulative += weight
                if r <= cumulative:
                    return self.variants[i]
                    
            # Fallback to last variant
            return self.variants[-1]
        
        # Default to first variant if strategy not implemented
        return self.variants[0]
    
    def log_impression(self, variant_id: str) -> bool:
        """
        Log an impression for a variant.
        
        Args:
            variant_id (str): The variant ID
            
        Returns:
            bool: True if successful
        """
        for variant in self.variants:
            if variant.id == variant_id:
                variant.impressions += 1
                return True
        return False
    
    def log_feedback(self, variant_id: str, is_positive: bool) -> bool:
        """
        Log feedback for a variant.
        
        Args:
            variant_id (str): The variant ID
            is_positive (bool): Whether feedback is positive
            
        Returns:
            bool: True if successful
        """
        for variant in self.variants:
            if variant.id == variant_id:
                if is_positive:
                    variant.positive_feedback += 1
                else:
                    variant.negative_feedback += 1
                return True
        return False

class ABTestingManager:
    """Manager for A/B testing experiments."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the A/B testing manager.
        
        Args:
            storage_path (Optional[str]): Path to store experiment data
        """
        self.storage_path = storage_path
        self.experiments: Dict[str, Experiment] = {}
        
        # Load experiments from storage if available
        if storage_path and os.path.exists(storage_path):
            self.load_experiments()
        
        logger.info("Initialized A/B testing manager")
    
    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        description: Optional[str] = None,
        allocation_strategy: AllocationStrategy = AllocationStrategy.EQUAL
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name (str): Experiment name
            variants (List[Dict]): List of variant configurations
            description (Optional[str]): Experiment description
            allocation_strategy (AllocationStrategy): How to allocate users
            
        Returns:
            Experiment: The created experiment
        """
        experiment_id = str(uuid.uuid4())
        
        # Create variant objects
        variant_objects = []
        for var in variants:
            variant_id = var.get('id', str(uuid.uuid4()))
            variant_objects.append(Variant(
                id=variant_id,
                name=var.get('name', f"Variant {len(variant_objects) + 1}"),
                model_id=var.get('model_id'),
                description=var.get('description'),
                weight=var.get('weight', 1.0)
            ))
        
        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            allocation_strategy=allocation_strategy,
            status=ExperimentStatus.DRAFT
        )
        
        # Add to registry
        self.experiments[experiment_id] = experiment
        
        # Save to storage
        self._save_experiments()
        
        logger.info(f"Created experiment '{name}' with {len(variant_objects)} variants")
        return experiment
    
    def start_experiment(self, experiment_id: str, duration_days: Optional[int] = None) -> bool:
        """
        Start an experiment.
        
        Args:
            experiment_id (str): Experiment ID
            duration_days (Optional[int]): Duration in days
            
        Returns:
            bool: True if successful
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
            
        experiment = self.experiments[experiment_id]
        
        # Set start date
        now = datetime.now()
        experiment.start_date = now.isoformat()
        
        # Set end date if duration specified
        if duration_days:
            end_date = now + timedelta(days=duration_days)
            experiment.end_date = end_date.isoformat()
        
        # Activate
        experiment.status = ExperimentStatus.ACTIVE
        
        # Save changes
        self._save_experiments()
        
        logger.info(f"Started experiment {experiment_id}")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop an experiment.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            bool: True if successful
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
            
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        
        # Save changes
        self._save_experiments()
        
        logger.info(f"Stopped experiment {experiment_id}")
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Optional[Experiment]: The experiment or None
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List[Dict]: List of experiment summaries
        """
        return [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "status": exp.status,
                "variant_count": len(exp.variants),
                "created_at": exp.created_at,
                "start_date": exp.start_date,
                "end_date": exp.end_date
            }
            for exp in self.experiments.values()
        ]
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get statistics for an experiment.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Dict: Experiment statistics
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return {"error": "Experiment not found"}
            
        experiment = self.experiments[experiment_id]
        
        # Calculate overall stats
        total_impressions = sum(v.impressions for v in experiment.variants)
        total_feedback = sum(v.total_feedback for v in experiment.variants)
        
        # Determine winning variant
        variants_with_feedback = [v for v in experiment.variants if v.total_feedback > 0]
        if variants_with_feedback:
            winner = max(variants_with_feedback, key=lambda v: v.success_rate)
            winner_id = winner.id
            confidence = min(1.0, winner.total_feedback / 100)  # Simple confidence heuristic
        else:
            winner_id = None
            confidence = 0
        
        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "is_active": experiment.is_active(),
            "total_impressions": total_impressions,
            "total_feedback": total_feedback,
            "variants": [v.to_dict() for v in experiment.variants],
            "winner_id": winner_id,
            "confidence": confidence,
            "start_date": experiment.start_date,
            "end_date": experiment.end_date
        }
    
    def select_variant(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Select a variant for a user.
        
        Args:
            experiment_id (str): Experiment ID
            user_id (str): User ID
            
        Returns:
            Optional[Dict]: Selected variant or None
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.is_active():
            return None
            
        variant = experiment.get_variant_for_user(user_id)
        if not variant:
            return None
            
        # Log impression
        experiment.log_impression(variant.id)
        self._save_experiments()
        
        return {
            "experiment_id": experiment_id,
            "variant_id": variant.id,
            "variant_name": variant.name,
            "model_id": variant.model_id
        }
    
    def record_feedback(
        self,
        experiment_id: str,
        variant_id: str,
        is_positive: bool
    ) -> bool:
        """
        Record feedback for a variant.
        
        Args:
            experiment_id (str): Experiment ID
            variant_id (str): Variant ID
            is_positive (bool): Whether feedback is positive
            
        Returns:
            bool: True if successful
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        success = experiment.log_feedback(variant_id, is_positive)
        if success:
            self._save_experiments()
            
        return success
    
    def load_experiments(self) -> None:
        """Load experiments from storage."""
        if not self.storage_path:
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            self.experiments = {}
            for exp_data in data:
                variants = [Variant(**v) for v in exp_data.pop('variants', [])]
                experiment = Experiment(variants=variants, **exp_data)
                self.experiments[experiment.id] = experiment
                
            logger.info(f"Loaded {len(self.experiments)} experiments from storage")
        except Exception as e:
            logger.error(f"Error loading experiments: {str(e)}")
    
    def _save_experiments(self) -> None:
        """Save experiments to storage."""
        if not self.storage_path:
            return
            
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Serialize experiments
            data = [exp.to_dict() for exp in self.experiments.values()]
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.experiments)} experiments to storage")
        except Exception as e:
            logger.error(f"Error saving experiments: {str(e)}") 