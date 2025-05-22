from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from app.utils.logger import setup_logger
from app.services.storage.dynamodb_service import DynamoDBService

logger = setup_logger(__name__)

class DataQualityService:
    """Service for ensuring data quality in user interactions."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        
    def validate_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single interaction record.
        
        Args:
            interaction: Interaction data to validate
            
        Returns:
            Dict containing validation results and cleaned data
        """
        results = {
            "is_valid": True,
            "issues": [],
            "cleaned_data": interaction.copy()
        }
        
        # Validate required fields
        required_fields = ["query", "response", "response_id"]
        for field in required_fields:
            if not interaction.get(field):
                results["is_valid"] = False
                results["issues"].append(f"Missing required field: {field}")
        
        # Validate query quality
        if "query" in interaction:
            query = interaction["query"]
            if len(query.strip()) < 3:
                results["is_valid"] = False
                results["issues"].append("Query too short")
            elif len(query.strip()) > 1000:
                results["is_valid"] = False
                results["issues"].append("Query too long")
            else:
                # Clean query
                results["cleaned_data"]["query"] = self._clean_text(query)
        
        # Validate response quality
        if "response" in interaction:
            response = interaction["response"]
            if len(response.strip()) < 5:
                results["is_valid"] = False
                results["issues"].append("Response too short")
            elif len(response.strip()) > 10000:
                results["is_valid"] = False
                results["issues"].append("Response too long")
            else:
                # Clean response
                results["cleaned_data"]["response"] = self._clean_text(response)
        
        # Validate timestamp
        if "timestamp" in interaction:
            try:
                ts = float(interaction["timestamp"])
                if ts > datetime.now().timestamp():
                    results["is_valid"] = False
                    results["issues"].append("Future timestamp")
            except (ValueError, TypeError):
                results["is_valid"] = False
                results["issues"].append("Invalid timestamp")
        
        # Validate feedback if present
        if "feedback" in interaction and interaction["feedback"]:
            feedback = interaction["feedback"]
            if len(feedback.strip()) > 1000:
                results["is_valid"] = False
                results["issues"].append("Feedback too long")
            else:
                results["cleaned_data"]["feedback"] = self._clean_text(feedback)
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove excessive punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        return text.strip()
    
    def check_data_quality(self, days: int = 7) -> Dict[str, Any]:
        """
        Check data quality metrics for recent interactions.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict containing data quality metrics
        """
        table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
        since = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        
        metrics = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "common_issues": {},
            "avg_query_length": 0,
            "avg_response_length": 0,
            "feedback_rate": 0
        }
        
        total_query_length = 0
        total_response_length = 0
        feedback_count = 0
        
        try:
            response = table.scan()
            items = response.get("Items", [])
            
            for item in items:
                metrics["total_records"] += 1
                
                # Skip old records
                if float(item.get("timestamp", 0)) < since:
                    continue
                
                # Validate record
                validation = self.validate_interaction(item)
                if validation["is_valid"]:
                    metrics["valid_records"] += 1
                    
                    # Update metrics
                    query_len = len(item.get("query", ""))
                    response_len = len(item.get("response", ""))
                    total_query_length += query_len
                    total_response_length += response_len
                    
                    if item.get("feedback"):
                        feedback_count += 1
                else:
                    metrics["invalid_records"] += 1
                    for issue in validation["issues"]:
                        metrics["common_issues"][issue] = metrics["common_issues"].get(issue, 0) + 1
            
            # Calculate averages
            if metrics["valid_records"] > 0:
                metrics["avg_query_length"] = total_query_length / metrics["valid_records"]
                metrics["avg_response_length"] = total_response_length / metrics["valid_records"]
                metrics["feedback_rate"] = feedback_count / metrics["valid_records"]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            raise
    
    def fix_data_quality_issues(self, days: int = 7) -> Dict[str, Any]:
        """
        Fix data quality issues in recent interactions.
        
        Args:
            days: Number of days to process
            
        Returns:
            Dict containing fix results
        """
        table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
        since = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        
        results = {
            "processed": 0,
            "fixed": 0,
            "failed": 0,
            "issues_fixed": {}
        }
        
        try:
            response = table.scan()
            items = response.get("Items", [])
            
            for item in items:
                results["processed"] += 1
                
                # Skip old records
                if float(item.get("timestamp", 0)) < since:
                    continue
                
                # Validate and clean record
                validation = self.validate_interaction(item)
                if not validation["is_valid"]:
                    try:
                        # Update record with cleaned data
                        table.update_item(
                            Key={"pk": item["pk"]},
                            UpdateExpression="SET #q = :q, #r = :r, #f = :f",
                            ExpressionAttributeNames={
                                "#q": "query",
                                "#r": "response",
                                "#f": "feedback"
                            },
                            ExpressionAttributeValues={
                                ":q": validation["cleaned_data"].get("query", ""),
                                ":r": validation["cleaned_data"].get("response", ""),
                                ":f": validation["cleaned_data"].get("feedback", "")
                            }
                        )
                        results["fixed"] += 1
                        for issue in validation["issues"]:
                            results["issues_fixed"][issue] = results["issues_fixed"].get(issue, 0) + 1
                    except Exception as e:
                        logger.error(f"Error fixing record {item['pk']}: {str(e)}")
                        results["failed"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error fixing data quality issues: {str(e)}")
            raise 