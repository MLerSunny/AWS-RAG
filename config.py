"""Configuration settings for the AWS RAG project."""
import os
from typing import Dict, Any, TypedDict, Literal, List
from exceptions import ConfigurationError, ErrorCode
from utils import safe_cast

class AWSConfig(TypedDict):
    """AWS configuration settings."""
    region: str
    s3_bucket: str
    dynamodb_table: str

class PathConfig(TypedDict):
    """Path configuration settings."""
    data_dir: str
    output_dir: str
    temp_dir: str

class APIConfig(TypedDict):
    """API configuration settings."""
    version: str
    timeout: int

class ProcessingConfig(TypedDict):
    """Processing configuration settings."""
    batch_size: int
    max_retries: int
    retry_delay: int

class LoggingConfig(TypedDict):
    """Logging configuration settings."""
    level: str
    format: str

class MessageConfig(TypedDict):
    """Message configuration settings."""
    errors: Dict[str, str]
    success: Dict[str, str]

class Config(TypedDict):
    """Complete configuration settings."""
    aws: AWSConfig
    paths: PathConfig
    api: APIConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    messages: MessageConfig

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'your-table-name')

# File paths
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
TEMP_DIR = 'temp'

# API Configuration
API_VERSION = 'v1'
API_TIMEOUT = 30  # seconds

# Processing Configuration
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Error Messages
ERROR_MESSAGES: Dict[str, str] = {
    'aws_connection': 'Failed to connect to AWS services',
    'file_not_found': 'File not found',
    'invalid_data': 'Invalid data format',
    'processing_error': 'Error during processing',
}

# Success Messages
SUCCESS_MESSAGES: Dict[str, str] = {
    'processing_complete': 'Processing completed successfully',
    'file_saved': 'File saved successfully',
    'data_updated': 'Data updated successfully',
}

def validate_aws_config() -> None:
    """Validate AWS configuration settings."""
    required_env_vars = ['AWS_REGION', 'S3_BUCKET', 'DYNAMODB_TABLE']
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}",
            context={'missing_vars': missing}
        )
        
    valid_regions = ['us-east-1', 'us-west-2', 'eu-west-1']  # Add more as needed
    if AWS_REGION not in valid_regions:
        raise ConfigurationError(
            f"Invalid AWS region: {AWS_REGION}",
            context={'region': AWS_REGION, 'valid_regions': valid_regions}
        )

def validate_paths() -> None:
    """Validate path configuration settings."""
    paths = [DATA_DIR, OUTPUT_DIR, TEMP_DIR]
    for path in paths:
        if not path or not isinstance(path, str):
            raise ConfigurationError(
                f"Invalid path configuration: {path}",
                context={'path': path}
            )

def validate_api_config() -> None:
    """Validate API configuration settings."""
    if not API_VERSION or not isinstance(API_VERSION, str):
        raise ConfigurationError(
            f"Invalid API version: {API_VERSION}",
            context={'version': API_VERSION}
        )
    if API_TIMEOUT <= 0:
        raise ConfigurationError(
            "API timeout must be positive",
            context={'timeout': API_TIMEOUT}
        )

def validate_processing_config() -> None:
    """Validate processing configuration settings."""
    if BATCH_SIZE <= 0:
        raise ConfigurationError(
            "Batch size must be positive",
            context={'batch_size': BATCH_SIZE}
        )
    if MAX_RETRIES < 0:
        raise ConfigurationError(
            "Max retries cannot be negative",
            context={'max_retries': MAX_RETRIES}
        )
    if RETRY_DELAY < 0:
        raise ConfigurationError(
            "Retry delay cannot be negative",
            context={'retry_delay': RETRY_DELAY}
        )

def validate_logging_config() -> None:
    """Validate logging configuration settings."""
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if LOG_LEVEL not in valid_levels:
        raise ConfigurationError(
            f"Invalid log level: {LOG_LEVEL}",
            context={'level': LOG_LEVEL, 'valid_levels': valid_levels}
        )
    if not LOG_FORMAT or not isinstance(LOG_FORMAT, str):
        raise ConfigurationError(
            f"Invalid log format: {LOG_FORMAT}",
            context={'format': LOG_FORMAT}
        )

def validate_config() -> None:
    """Validate all configuration settings.
    
    Raises:
        ConfigurationError: If any configuration is invalid
    """
    validate_aws_config()
    validate_paths()
    validate_api_config()
    validate_processing_config()
    validate_logging_config()

def get_config() -> Config:
    """Get all configuration settings.
    
    Returns:
        Config: Complete configuration settings
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    validate_config()
    
    return Config(
        aws=AWSConfig(
            region=AWS_REGION,
            s3_bucket=S3_BUCKET,
            dynamodb_table=DYNAMODB_TABLE,
        ),
        paths=PathConfig(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            temp_dir=TEMP_DIR,
        ),
        api=APIConfig(
            version=API_VERSION,
            timeout=API_TIMEOUT,
        ),
        processing=ProcessingConfig(
            batch_size=BATCH_SIZE,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
        ),
        logging=LoggingConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
        ),
        messages=MessageConfig(
            errors=ERROR_MESSAGES,
            success=SUCCESS_MESSAGES,
        )
    ) 