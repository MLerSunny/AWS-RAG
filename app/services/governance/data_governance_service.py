from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import re
from decimal import Decimal
from app.utils.logger import setup_logger
from app.services.storage.dynamodb_service import DynamoDBService

logger = setup_logger(__name__)

class DataGovernanceService:
    """Service for data governance, retention, and privacy."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        
        # Default retention periods (in days)
        self.retention_periods = {
            "user_interactions": 90,  # 3 months
            "feedback": 365,  # 1 year
            "model_metrics": 730,  # 2 years
        }
        
        # Privacy settings
        self.privacy_settings = {
            "pii_fields": ["user_id", "email", "ip_address"],
            "mask_patterns": {
                "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "ip": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            }
        }
    
    def apply_data_retention(self) -> Dict[str, Any]:
        """
        Apply data retention policies to stored data.
        
        Returns:
            Dict containing retention results
        """
        results = {
            "processed": 0,
            "deleted": 0,
            "failed": 0,
            "tables": {}
        }
        
        for table_name, retention_days in self.retention_periods.items():
            try:
                table = self.db_service._get_table(f"{self.table_prefix}{table_name}")
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                cutoff_ts = Decimal(str(cutoff_date.timestamp()))
                
                # Scan for old records
                response = table.scan(
                    FilterExpression="timestamp < :ts",
                    ExpressionAttributeValues={":ts": cutoff_ts}
                )
                old_items = response.get("Items", [])
                
                # Delete old records
                deleted = 0
                failed = 0
                for item in old_items:
                    try:
                        table.delete_item(Key={"pk": item["pk"]})
                        deleted += 1
                    except Exception as e:
                        logger.error(f"Error deleting item {item['pk']}: {str(e)}")
                        failed += 1
                
                results["tables"][table_name] = {
                    "processed": len(old_items),
                    "deleted": deleted,
                    "failed": failed
                }
                results["processed"] += len(old_items)
                results["deleted"] += deleted
                results["failed"] += failed
                
            except Exception as e:
                logger.error(f"Error processing table {table_name}: {str(e)}")
                results["tables"][table_name] = {"error": str(e)}
        
        return results
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize sensitive data in a record.
        
        Args:
            data: Data to anonymize
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Anonymize PII fields
        for field in self.privacy_settings["pii_fields"]:
            if field in anonymized:
                value = anonymized[field]
                if isinstance(value, str):
                    # Hash the value
                    anonymized[field] = hashlib.sha256(value.encode()).hexdigest()
        
        # Mask patterns in text fields
        for field, value in anonymized.items():
            if isinstance(value, str):
                # Mask email addresses
                if "email" in self.privacy_settings["mask_patterns"]:
                    value = re.sub(
                        self.privacy_settings["mask_patterns"]["email"],
                        "[EMAIL]",
                        value
                    )
                # Mask IP addresses
                if "ip" in self.privacy_settings["mask_patterns"]:
                    value = re.sub(
                        self.privacy_settings["mask_patterns"]["ip"],
                        "[IP]",
                        value
                    )
                anonymized[field] = value
        
        return anonymized
    
    def check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data compliance with privacy regulations.
        
        Args:
            data: Data to check
            
        Returns:
            Dict containing compliance check results
        """
        results = {
            "is_compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for PII
        for field in self.privacy_settings["pii_fields"]:
            if field in data:
                results["is_compliant"] = False
                results["issues"].append(f"Contains PII field: {field}")
                results["recommendations"].append(f"Anonymize {field}")
        
        # Check for sensitive patterns
        for field, value in data.items():
            if isinstance(value, str):
                # Check for email addresses
                if "email" in self.privacy_settings["mask_patterns"]:
                    if re.search(self.privacy_settings["mask_patterns"]["email"], value):
                        results["is_compliant"] = False
                        results["issues"].append(f"Contains email address in {field}")
                        results["recommendations"].append(f"Mask email in {field}")
                
                # Check for IP addresses
                if "ip" in self.privacy_settings["mask_patterns"]:
                    if re.search(self.privacy_settings["mask_patterns"]["ip"], value):
                        results["is_compliant"] = False
                        results["issues"].append(f"Contains IP address in {field}")
                        results["recommendations"].append(f"Mask IP in {field}")
        
        return results
    
    def get_data_lineage(self, record_id: str) -> Dict[str, Any]:
        """
        Get data lineage for a record.
        
        Args:
            record_id: ID of the record
            
        Returns:
            Dict containing lineage information
        """
        lineage = {
            "record_id": record_id,
            "created_at": None,
            "modified_at": None,
            "transformations": [],
            "access_log": []
        }
        
        try:
            # Get record from user interactions table
            table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
            response = table.get_item(Key={"pk": record_id})
            item = response.get("Item")
            
            if item:
                lineage["created_at"] = item.get("timestamp")
                lineage["modified_at"] = item.get("last_modified")
                
                # Get transformations
                if "transformations" in item:
                    lineage["transformations"] = item["transformations"]
                
                # Get access log
                if "access_log" in item:
                    lineage["access_log"] = item["access_log"]
            
            return lineage
            
        except Exception as e:
            logger.error(f"Error getting data lineage: {str(e)}")
            raise
    
    def update_retention_policy(self, table_name: str, days: int) -> bool:
        """
        Update retention period for a table.
        
        Args:
            table_name: Name of the table
            days: New retention period in days
            
        Returns:
            True if update was successful
        """
        if table_name not in self.retention_periods:
            logger.error(f"Unknown table: {table_name}")
            return False
        
        if days < 1:
            logger.error("Retention period must be at least 1 day")
            return False
        
        self.retention_periods[table_name] = days
        logger.info(f"Updated retention period for {table_name} to {days} days")
        return True
    
    def update_privacy_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update privacy settings.
        
        Args:
            settings: New privacy settings
            
        Returns:
            True if update was successful
        """
        if not isinstance(settings, dict):
            logger.error("Settings must be a dictionary")
            return False
        
        # Validate settings
        if "pii_fields" in settings:
            if not isinstance(settings["pii_fields"], list):
                logger.error("pii_fields must be a list")
                return False
            self.privacy_settings["pii_fields"] = settings["pii_fields"]
        
        if "mask_patterns" in settings:
            if not isinstance(settings["mask_patterns"], dict):
                logger.error("mask_patterns must be a dictionary")
                return False
            self.privacy_settings["mask_patterns"].update(settings["mask_patterns"])
        
        logger.info("Updated privacy settings")
        return True 