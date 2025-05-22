from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import boto3
from app.utils.logger import setup_logger
from app.services.quality.data_quality_service import DataQualityService
from app.services.storage.dynamodb_service import DynamoDBService

logger = setup_logger(__name__)

class MonitoringService:
    """Service for monitoring system health and data quality."""
    
    def __init__(self, table_prefix: str = "genai_"):
        self.table_prefix = table_prefix
        self.db_service = DynamoDBService(table_prefix=table_prefix)
        self.data_quality = DataQualityService(table_prefix=table_prefix)
        self.cloudwatch = boto3.client('cloudwatch')
        
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.
        
        Returns:
            Dict containing health metrics
        """
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check DynamoDB connection
        try:
            table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
            table.scan(Limit=1)
            health["checks"]["dynamodb"] = "healthy"
        except Exception as e:
            health["checks"]["dynamodb"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        # Check data quality
        try:
            quality_metrics = self.data_quality.check_data_quality(days=1)
            health["checks"]["data_quality"] = {
                "status": "healthy" if quality_metrics["valid_records"] > 0 else "degraded",
                "metrics": quality_metrics
            }
        except Exception as e:
            health["checks"]["data_quality"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        # Check CloudWatch metrics
        try:
            metrics = self._get_cloudwatch_metrics()
            health["checks"]["cloudwatch"] = {
                "status": "healthy",
                "metrics": metrics
            }
        except Exception as e:
            health["checks"]["cloudwatch"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        return health
    
    def _get_cloudwatch_metrics(self) -> Dict[str, Any]:
        """
        Get relevant CloudWatch metrics.
        
        Returns:
            Dict containing CloudWatch metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        metrics = {}
        
        # Get DynamoDB metrics
        dynamodb_metrics = [
            "ConsumedWriteCapacityUnits",
            "ConsumedReadCapacityUnits",
            "ThrottledRequests"
        ]
        
        for metric in dynamodb_metrics:
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/DynamoDB",
                MetricName=metric,
                Dimensions=[{"Name": "TableName", "Value": f"{self.table_prefix}user_interactions"}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=["Sum"]
            )
            metrics[metric] = response.get("Datapoints", [])
        
        return metrics
    
    def check_data_pipeline(self) -> Dict[str, Any]:
        """
        Check data pipeline health.
        
        Returns:
            Dict containing pipeline metrics
        """
        pipeline = {
            "status": "healthy",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check recent data ingestion
        try:
            table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
            response = table.scan(
                FilterExpression="timestamp > :ts",
                ExpressionAttributeValues={":ts": (datetime.utcnow() - timedelta(hours=1)).timestamp()}
            )
            recent_records = len(response.get("Items", []))
            pipeline["metrics"]["recent_records"] = recent_records
            
            if recent_records == 0:
                pipeline["status"] = "warning"
                pipeline["issues"] = ["No recent data ingestion"]
        except Exception as e:
            pipeline["status"] = "degraded"
            pipeline["issues"] = [f"Error checking data ingestion: {str(e)}"]
        
        # Check data quality
        try:
            quality_metrics = self.data_quality.check_data_quality(days=1)
            pipeline["metrics"]["data_quality"] = quality_metrics
            
            if quality_metrics["invalid_records"] > 0:
                pipeline["status"] = "warning"
                if "issues" not in pipeline:
                    pipeline["issues"] = []
                pipeline["issues"].append("Data quality issues detected")
        except Exception as e:
            pipeline["status"] = "degraded"
            if "issues" not in pipeline:
                pipeline["issues"] = []
            pipeline["issues"].append(f"Error checking data quality: {str(e)}")
        
        return pipeline
    
    def send_alert(self, alert_type: str, message: str, severity: str = "warning") -> bool:
        """
        Send an alert about system issues.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error)
            
        Returns:
            True if alert was sent successfully
        """
        try:
            # Create CloudWatch alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{self.table_prefix}{alert_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                MetricName="SystemHealth",
                Namespace="Custom/GenAI",
                Statistic="Average",
                Period=300,
                EvaluationPeriods=1,
                Threshold=1.0,
                ComparisonOperator="GreaterThanThreshold",
                Dimensions=[
                    {"Name": "AlertType", "Value": alert_type},
                    {"Name": "Severity", "Value": severity}
                ],
                AlarmDescription=message
            )
            
            # Log alert
            logger.warning(f"Alert sent - Type: {alert_type}, Severity: {severity}, Message: {message}")
            
            return True
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
            return False
    
    def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get system metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing system metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = {
            "timestamp": end_time.isoformat(),
            "period_hours": hours,
            "data_metrics": {},
            "system_metrics": {}
        }
        
        # Get data metrics
        try:
            table = self.db_service._get_table(f"{self.table_prefix}user_interactions")
            response = table.scan()
            items = response.get("Items", [])
            
            # Filter items by time range
            filtered_items = [
                item for item in items
                if float(item.get("timestamp", 0)) >= start_time.timestamp()
            ]
            
            metrics["data_metrics"] = {
                "total_records": len(filtered_items),
                "records_per_hour": len(filtered_items) / hours,
                "feedback_rate": sum(1 for item in filtered_items if item.get("feedback")) / len(filtered_items) if filtered_items else 0
            }
        except Exception as e:
            logger.error(f"Error getting data metrics: {str(e)}")
            metrics["data_metrics"]["error"] = str(e)
        
        # Get system metrics from CloudWatch
        try:
            cloudwatch_metrics = self._get_cloudwatch_metrics()
            metrics["system_metrics"] = cloudwatch_metrics
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            metrics["system_metrics"]["error"] = str(e)
        
        return metrics 