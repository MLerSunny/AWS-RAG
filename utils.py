"""Utility functions for the AWS RAG project."""
import os
import json
import boto3
from typing import Dict, List, Optional, Any, TypeVar, cast, Union
from datetime import datetime
import logging
from exceptions import (
    FileOperationError, AWSServiceError, ErrorCode,
    ValidationError, ProcessingError
)
from logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

def get_aws_client(service_name: str, region_name: str = 'us-east-1') -> Any:
    """Get AWS client for specified service.
    
    Args:
        service_name: Name of the AWS service
        region_name: AWS region name
        
    Returns:
        Any: AWS service client
        
    Raises:
        AWSServiceError: If client creation fails
    """
    try:
        return boto3.client(service_name, region_name=region_name)
    except Exception as e:
        raise AWSServiceError(
            service_name,
            'client_creation',
            str(e),
            ErrorCode.AWS_INIT_FAILED,
            cause=e
        )

def get_aws_resource(service_name: str, region_name: str = 'us-east-1') -> Any:
    """Get AWS resource for specified service.
    
    Args:
        service_name: Name of the AWS service
        region_name: AWS region name
        
    Returns:
        Any: AWS service resource
        
    Raises:
        AWSServiceError: If resource creation fails
    """
    try:
        return boto3.resource(service_name, region_name=region_name)
    except Exception as e:
        raise AWSServiceError(
            service_name,
            'resource_creation',
            str(e),
            ErrorCode.AWS_INIT_FAILED,
            cause=e
        )

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict[str, Any]: Parsed JSON data
        
    Raises:
        FileOperationError: If file doesn't exist or contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileOperationError(
            'read',
            file_path,
            f"File not found: {str(e)}",
            ErrorCode.FILE_NOT_FOUND,
            cause=e
        )
    except json.JSONDecodeError as e:
        raise FileOperationError(
            'read',
            file_path,
            f"Invalid JSON: {str(e)}",
            ErrorCode.DATA_FORMAT_ERROR,
            cause=e
        )
    except Exception as e:
        raise FileOperationError(
            'read',
            file_path,
            str(e),
            ErrorCode.FILE_READ_ERROR,
            cause=e
        )

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        
    Raises:
        FileOperationError: If file cannot be written
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise FileOperationError(
            'write',
            file_path,
            str(e),
            ErrorCode.FILE_WRITE_ERROR,
            cause=e
        )

def get_timestamp() -> str:
    """Get current timestamp in ISO format.
    
    Returns:
        str: ISO formatted timestamp
    """
    return datetime.utcnow().isoformat()

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Raises:
        FileOperationError: If directory cannot be created
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        raise FileOperationError(
            'create',
            directory,
            str(e),
            ErrorCode.FILE_ACCESS_DENIED,
            cause=e
        )

def format_error_message(error: Exception) -> str:
    """Format error message for logging.
    
    Args:
        error: Exception to format
        
    Returns:
        str: Formatted error message
    """
    if isinstance(error, AWSServiceError):
        return f"[{error.error_code}] {error.message}"
    return f"{type(error).__name__}: {str(error)}"

def safe_cast(value: Any, target_type: type[T]) -> T:
    """Safely cast a value to a target type.
    
    Args:
        value: Value to cast
        target_type: Target type
        
    Returns:
        T: Casted value
        
    Raises:
        ValidationError: If value cannot be cast to target type
    """
    try:
        return cast(T, value)
    except Exception as e:
        raise ValidationError(
            f"Cannot cast {value} to {target_type.__name__}",
            context={'value': value, 'target_type': target_type.__name__},
            cause=e
        )

def validate_dict_structure(data: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate dictionary structure.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        
    Raises:
        ValidationError: If dictionary is invalid
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(
            f"Missing required keys: {', '.join(missing_keys)}",
            context={'missing_keys': missing_keys}
        ) 