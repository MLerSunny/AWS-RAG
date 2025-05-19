from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Callable, cast, List
import boto3
from botocore.exceptions import ClientError
import logging
import time
from config import AWS_REGION
from exceptions import AWSServiceError, ErrorCode
from logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

def get_aws_client(service_name: str, region_name: str = AWS_REGION) -> Any:
    """Get AWS client for specified service."""
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

def get_aws_resource(service_name: str, region_name: str = AWS_REGION) -> Any:
    """Get AWS resource for specified service."""
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

def format_error_message(error: Exception) -> str:
    """Format error message for logging."""
    if isinstance(error, AWSServiceError):
        return f"[{error.error_code}] {error.message}"
    return f"{type(error).__name__}: {str(error)}"

class AWSBase(ABC):
    """Base class for AWS service interactions."""
    
    def __init__(self, service_name: str, region_name: str = AWS_REGION) -> None:
        """Initialize AWS service client and resource.
        
        Args:
            service_name: Name of the AWS service
            region_name: AWS region name
            
        Raises:
            AWSServiceError: If client or resource creation fails
        """
        self.service_name = service_name
        self.region_name = region_name
        try:
            self.client = get_aws_client(service_name, region_name)
            self.resource = get_aws_resource(service_name, region_name)
        except Exception as e:
            raise AWSServiceError(
                service_name,
                'initialization',
                str(e),
                ErrorCode.AWS_INIT_FAILED,
                cause=e
            )
        
    def handle_aws_error(self, error: ClientError, operation: str) -> None:
        """Handle AWS client errors with proper logging.
        
        Args:
            error: AWS client error
            operation: Name of the operation that failed
            
        Raises:
            AWSServiceError: Wrapped AWS client error
        """
        error_message = format_error_message(error)
        logger.error(f"AWS {self.service_name} {operation} failed: {error_message}")
        
        # Map AWS error codes to our error codes
        error_code = ErrorCode.AWS_OPERATION_FAILED
        if 'AccessDenied' in str(error):
            error_code = ErrorCode.AWS_PERMISSION_DENIED
        elif 'NoSuchKey' in str(error) or 'NoSuchBucket' in str(error):
            error_code = ErrorCode.AWS_RESOURCE_NOT_FOUND
            
        raise AWSServiceError(
            self.service_name,
            operation,
            error_message,
            error_code,
            cause=error
        )
        
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate AWS credentials.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            AWSServiceError: If credentials are invalid
        """
        pass
        
    @abstractmethod
    def check_permissions(self) -> bool:
        """Check required permissions.
        
        Returns:
            bool: True if all required permissions are available
            
        Raises:
            AWSServiceError: If required permissions are missing
        """
        pass
        
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information.
        
        Returns:
            Dict[str, Any]: Service status information
            
        Raises:
            AWSServiceError: If status check fails
        """
        try:
            return {
                'service': self.service_name,
                'region': self.region_name,
                'status': 'active'
            }
        except Exception as e:
            error_message = format_error_message(e)
            logger.error(f"Error getting service status: {error_message}")
            return {
                'service': self.service_name,
                'region': self.region_name,
                'status': 'error',
                'error': error_message
            }
            
    def retry_operation(self, operation: Callable[[], T], max_retries: int = 3) -> T:
        """Retry an operation with exponential backoff.
        
        Args:
            operation: Callable that returns type T
            max_retries: Maximum number of retry attempts
            
        Returns:
            T: Result of the operation
            
        Raises:
            AWSServiceError: If all retry attempts fail
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                return operation()
            except ClientError as e:
                last_error = e
                retries += 1
                if retries == max_retries:
                    break
                time.sleep(2 ** retries)  # Exponential backoff
                
        if last_error:
            self.handle_aws_error(last_error, 'retry_operation')
        raise AWSServiceError(
            self.service_name,
            'retry_operation',
            'Operation failed after all retries',
            ErrorCode.AWS_OPERATION_FAILED,
            cause=last_error
        )
                
    def get_resource_arn(self, resource_id: str) -> str:
        """Get ARN for a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            str: Resource ARN
        """
        return f"arn:aws:{self.service_name}:{self.region_name}:{resource_id}"
        
    def validate_aws_response(self, response: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate AWS API response.
        
        Args:
            response: AWS API response
            required_fields: List of required field names
            
        Raises:
            AWSServiceError: If response is invalid
        """
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise AWSServiceError(
                self.service_name,
                'validate_response',
                f"Missing required fields in response: {', '.join(missing_fields)}",
                ErrorCode.VALIDATION_FAILED,
                context={'missing_fields': missing_fields}
            )
            
    def check_aws_error_response(self, response: Dict[str, Any]) -> None:
        """Check AWS response for error conditions.
        
        Args:
            response: AWS API response
            
        Raises:
            AWSServiceError: If response indicates an error
        """
        if 'Error' in response:
            error = response['Error']
            error_code = error.get('Code', 'UnknownError')
            error_message = error.get('Message', 'Unknown error occurred')
            raise AWSServiceError(
                self.service_name,
                'api_call',
                error_message,
                ErrorCode.AWS_OPERATION_FAILED,
                context={'error_code': error_code}
            ) 