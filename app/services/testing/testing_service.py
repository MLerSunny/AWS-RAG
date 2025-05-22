from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import random
from app.utils.logger import setup_logger
from app.services.storage.dynamodb_service import DynamoDBService
from app.services.quality.data_quality_service import DataQualityService
from app.services.models.model_management_service import ModelManagementService

logger = setup_logger(__name__)

class TestingService:
    """Service for automated testing and validation."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        self.data_quality = DataQualityService(table_prefix=table_prefix)
        self.model_management = ModelManagementService(table_prefix=table_prefix)
        
    def run_data_quality_tests(self) -> Dict[str, Any]:
        """
        Run data quality test suite.
        
        Returns:
            Dict containing test results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": {},
            "overall_status": "passed"
        }
        
        # Test data validation
        try:
            validation_results = self.data_quality.check_data_quality(days=7)
            results["tests"]["data_validation"] = {
                "status": "passed" if validation_results["valid_records"] > 0 else "failed",
                "details": validation_results
            }
        except Exception as e:
            results["tests"]["data_validation"] = {
                "status": "error",
                "error": str(e)
            }
            results["overall_status"] = "error"
        
        # Test data cleaning
        try:
            test_data = {
                "query": "Test query with email@example.com and IP 192.168.1.1",
                "response": "Test response",
                "user_id": "test_user"
            }
            cleaned_data = self.data_quality.validate_interaction(test_data)
            results["tests"]["data_cleaning"] = {
                "status": "passed" if cleaned_data["is_valid"] else "failed",
                "details": cleaned_data
            }
        except Exception as e:
            results["tests"]["data_cleaning"] = {
                "status": "error",
                "error": str(e)
            }
            results["overall_status"] = "error"
        
        return results
    
    def run_model_tests(self, model_id: str) -> Dict[str, Any]:
        """
        Run model test suite.
        
        Args:
            model_id: ID of the model to test
            
        Returns:
            Dict containing test results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "tests": {},
            "overall_status": "passed"
        }
        
        # Test model metrics
        try:
            metrics = self.model_management.get_model_metrics(model_id)
            results["tests"]["metrics"] = {
                "status": "passed" if metrics else "failed",
                "details": metrics
            }
        except Exception as e:
            results["tests"]["metrics"] = {
                "status": "error",
                "error": str(e)
            }
            results["overall_status"] = "error"
        
        # Test model deployment
        try:
            deployment_status = self.model_management.deploy_model(model_id, "testing")
            results["tests"]["deployment"] = {
                "status": "passed" if deployment_status else "failed",
                "details": {"deployed": deployment_status}
            }
        except Exception as e:
            results["tests"]["deployment"] = {
                "status": "error",
                "error": str(e)
            }
            results["overall_status"] = "error"
        
        return results
    
    def run_ab_test(self, experiment_id: str, duration_days: int = 7) -> Dict[str, Any]:
        """
        Run A/B test for model variants.
        
        Args:
            experiment_id: ID of the experiment
            duration_days: Duration of the test in days
            
        Returns:
            Dict containing test results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": experiment_id,
            "duration_days": duration_days,
            "variants": {},
            "overall_status": "running"
        }
        
        try:
            # Get experiment data
            table = self.db_service._get_table(f"{self.table_prefix}experiments")
            response = table.get_item(Key={"id": experiment_id})
            experiment = response.get("Item")
            
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Get variant performance
            for variant in experiment.get("variants", []):
                variant_id = variant["id"]
                model_id = variant["model_id"]
                
                # Get model metrics
                metrics = self.model_management.get_model_metrics(model_id, days=duration_days)
                
                results["variants"][variant_id] = {
                    "model_id": model_id,
                    "metrics": metrics,
                    "status": "active"
                }
            
            # Determine winner
            if all(v["status"] == "active" for v in results["variants"].values()):
                winner = self._determine_winner(results["variants"])
                if winner:
                    results["winner"] = winner
                    results["overall_status"] = "completed"
            
            return results
            
        except Exception as e:
            logger.error(f"Error running A/B test: {str(e)}")
            results["overall_status"] = "error"
            results["error"] = str(e)
            return results
    
    def _determine_winner(self, variants: Dict[str, Any]) -> Optional[str]:
        """
        Determine the winning variant based on metrics.
        
        Args:
            variants: Dict of variant data
            
        Returns:
            ID of the winning variant or None
        """
        try:
            if not isinstance(variants, dict):
                return None
                
            # Calculate scores for each variant
            scores = {}
            for variant_id, data in variants.items():
                if not isinstance(variant_id, str) or not isinstance(data, dict):
                    continue
                    
                metrics = data.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                    
                current_metrics = metrics.get("current_metrics", {})
                if not isinstance(current_metrics, dict):
                    continue
                
                # Calculate score based on key metrics
                score = 0
                if "accuracy" in current_metrics and isinstance(current_metrics["accuracy"], (int, float)):
                    score += float(current_metrics["accuracy"]) * 0.4
                if "response_time" in current_metrics and isinstance(current_metrics["response_time"], (int, float)):
                    score += (1 / float(current_metrics["response_time"])) * 0.3
                if "user_satisfaction" in current_metrics and isinstance(current_metrics["user_satisfaction"], (int, float)):
                    score += float(current_metrics["user_satisfaction"]) * 0.3
                
                scores[variant_id] = score
            
            # Find winner
            if scores:
                winner = max(scores.items(), key=lambda x: x[1])
                return winner[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining winner: {str(e)}")
            return None
    
    def generate_test_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """
        Generate test data for validation.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            List of test records
        """
        test_data = []
        
        # Sample queries and responses
        queries = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain quantum computing",
            "What is machine learning?",
            "Tell me about AWS services"
        ]
        
        responses = [
            "The capital of France is Paris.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Quantum computing uses quantum bits to perform calculations.",
            "Machine learning is a subset of AI that focuses on training models.",
            "AWS offers various cloud computing services."
        ]
        
        for _ in range(num_records):
            # Generate random data
            query = random.choice(queries)
            response = random.choice(responses)
            user_id = f"test_user_{random.randint(1, 1000)}"
            timestamp = datetime.utcnow().timestamp() - random.randint(0, 86400)
            
            # Create test record
            record = {
                "query": query,
                "response": response,
                "user_id": user_id,
                "timestamp": timestamp,
                "is_helpful": random.choice([True, False]),
                "feedback": random.choice(["", "Good response", "Not helpful", "Could be better"])
            }
            
            test_data.append(record)
        
        return test_data
    
    def validate_test_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate test results.
        
        Args:
            results: Test results to validate
            
        Returns:
            True if results are valid
        """
        try:
            # Check required fields
            required_fields = ["timestamp", "tests", "overall_status"]
            if not all(field in results for field in required_fields):
                return False
            
            # Validate timestamp
            try:
                datetime.fromisoformat(results["timestamp"])
            except:
                return False
            
            # Validate test results
            for test_name, test_result in results["tests"].items():
                if "status" not in test_result:
                    return False
                if test_result["status"] not in ["passed", "failed", "error"]:
                    return False
            
            # Validate overall status
            if results["overall_status"] not in ["passed", "failed", "error", "running", "completed"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating test results: {str(e)}")
            return False 