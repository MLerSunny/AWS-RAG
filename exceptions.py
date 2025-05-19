"""Custom exceptions for the AWS RAG project."""
from typing import Optional, Dict, Any

class ErrorCode:
    """Error codes for AWS RAG project."""
    # Configuration errors (1000-1999)
    CONFIG_MISSING = 1000
    CONFIG_INVALID = 1001
    ENV_VAR_MISSING = 1002
    CONFIG_VALIDATION_FAILED = 1003
    
    # AWS service errors (2000-2999)
    AWS_INIT_FAILED = 2000
    AWS_OPERATION_FAILED = 2001
    AWS_PERMISSION_DENIED = 2002
    AWS_RESOURCE_NOT_FOUND = 2003
    AWS_RATE_LIMIT_EXCEEDED = 2004
    AWS_SERVICE_UNAVAILABLE = 2005
    AWS_INVALID_CREDENTIALS = 2006
    
    # File operation errors (3000-3999)
    FILE_NOT_FOUND = 3000
    FILE_ACCESS_DENIED = 3001
    FILE_READ_ERROR = 3002
    FILE_WRITE_ERROR = 3003
    FILE_DELETE_ERROR = 3004
    FILE_COPY_ERROR = 3005
    FILE_MOVE_ERROR = 3006
    
    # Processing errors (4000-4999)
    PROCESSING_FAILED = 4000
    VALIDATION_FAILED = 4001
    DATA_FORMAT_ERROR = 4002
    DATA_TRANSFORMATION_ERROR = 4003
    DATA_INTEGRITY_ERROR = 4004
    DATA_CONSISTENCY_ERROR = 4005
    
    # Network errors (5000-5999)
    NETWORK_ERROR = 5000
    TIMEOUT_ERROR = 5001
    CONNECTION_ERROR = 5002
    DNS_ERROR = 5003
    
    # Security errors (6000-6999)
    AUTHENTICATION_ERROR = 6000
    AUTHORIZATION_ERROR = 6001
    TOKEN_ERROR = 6002
    ENCRYPTION_ERROR = 6003

class AWSRAGError(Exception):
    """Base exception for AWS RAG project."""
    def __init__(
        self,
        message: str,
        error_code: int,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        """Initialize error.
        
        Args:
            message: Error message
            error_code: Error code
            context: Additional error context
            cause: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        super().__init__(message, *args)
        
    def __str__(self) -> str:
        """Get string representation of error."""
        result = f"[{self.error_code}] {self.message}"
        if self.context:
            result += f" (Context: {self.context})"
        if self.cause:
            result += f" (Caused by: {str(self.cause)})"
        return result

class ConfigurationError(AWSRAGError):
    """Configuration related errors."""
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        super().__init__(message, ErrorCode.CONFIG_INVALID, context, cause, *args)

class AWSServiceError(AWSRAGError):
    """AWS service related errors."""
    def __init__(
        self,
        service: str,
        operation: str,
        message: str,
        error_code: int = ErrorCode.AWS_OPERATION_FAILED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        self.service = service
        self.operation = operation
        context = context or {}
        context.update({
            'service': service,
            'operation': operation
        })
        super().__init__(
            f"AWS {service} {operation} failed: {message}",
            error_code,
            context,
            cause,
            *args
        )

class ProcessingError(AWSRAGError):
    """Data processing related errors."""
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        super().__init__(message, ErrorCode.PROCESSING_FAILED, context, cause, *args)

class ValidationError(AWSRAGError):
    """Data validation related errors."""
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        super().__init__(message, ErrorCode.VALIDATION_FAILED, context, cause, *args)

class FileOperationError(AWSRAGError):
    """File operation related errors."""
    def __init__(
        self,
        operation: str,
        file_path: str,
        message: str,
        error_code: int = ErrorCode.FILE_READ_ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        self.operation = operation
        self.file_path = file_path
        context = context or {}
        context.update({
            'operation': operation,
            'file_path': file_path
        })
        super().__init__(
            f"File {operation} failed for {file_path}: {message}",
            error_code,
            context,
            cause,
            *args
        )

class NetworkError(AWSRAGError):
    """Network related errors."""
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.NETWORK_ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        super().__init__(message, error_code, context, cause, *args)

class SecurityError(AWSRAGError):
    """Security related errors."""
    def __init__(
        self,
        message: str,
        error_code: int = ErrorCode.AUTHENTICATION_ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        *args: object
    ) -> None:
        super().__init__(message, error_code, context, cause, *args) 