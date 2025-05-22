from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Type
from datetime import datetime
from decimal import Decimal
import json
from pydantic import BaseModel, Field, validator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class DynamoDBTypeConverter:
    """Utility class for converting DynamoDB types to Python types and vice versa."""
    
    @staticmethod
    def to_python(value: Any) -> Any:
        """
        Convert DynamoDB type to Python type.
        
        Args:
            value: DynamoDB value to convert
            
        Returns:
            Converted Python value
        """
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, dict):
            return {k: DynamoDBTypeConverter.to_python(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [DynamoDBTypeConverter.to_python(item) for item in value]
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            logger.warning(f"Unexpected type in DynamoDB response: {type(value)}")
            return str(value)
    
    @staticmethod
    def to_dynamodb(value: Any) -> Any:
        """
        Convert Python type to DynamoDB type.
        
        Args:
            value: Python value to convert
            
        Returns:
            Converted DynamoDB value
        """
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, dict):
            return {k: DynamoDBTypeConverter.to_dynamodb(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [DynamoDBTypeConverter.to_dynamodb(item) for item in value]
        elif isinstance(value, (str, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            return value.timestamp()
        else:
            logger.warning(f"Unexpected type for DynamoDB conversion: {type(value)}")
            return str(value)

class DynamoDBItem(BaseModel, Generic[T]):
    """Base model for DynamoDB items with type conversion."""
    
    @classmethod
    def from_dynamodb(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create instance from DynamoDB response.
        
        Args:
            data: Raw DynamoDB response data
            
        Returns:
            DynamoDBItem instance
        """
        converted_data = DynamoDBTypeConverter.to_python(data)
        return cls(**converted_data)
    
    def to_dynamodb(self) -> Dict[str, Any]:
        """
        Convert instance to DynamoDB format.
        
        Returns:
            Dict in DynamoDB format
        """
        return DynamoDBTypeConverter.to_dynamodb(self.dict())

class UserInteraction(DynamoDBItem['UserInteraction']):
    """Model for user interaction data."""
    
    pk: str = Field(..., description="Primary key (response_id)")
    response_id: str = Field(..., description="Unique identifier for the response")
    query: str = Field(..., description="User query")
    response: str = Field(..., description="System response")
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model_id: Optional[str] = None
    is_helpful: Optional[bool] = None
    feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is a positive number."""
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and within length limits."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query exceeds maximum length of 1000 characters")
        return v
    
    @validator('response')
    def validate_response(cls, v):
        """Validate response is not empty."""
        if not v.strip():
            raise ValueError("Response cannot be empty")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata is JSON serializable."""
        if v is not None:
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {str(e)}")
        return v 