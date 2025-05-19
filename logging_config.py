"""Logging configuration for the AWS RAG project."""
import logging.config
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
from config import LOG_LEVEL, LOG_FORMAT
from exceptions import ConfigurationError, ErrorCode

VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
MAX_LOG_FILE_SIZE = 10485760  # 10MB
MAX_BACKUP_COUNT = 5
MIN_BACKUP_COUNT = 1
DEFAULT_LOG_DIR = 'logs'
DEFAULT_LOG_FILE = 'aws_rag.log'

def validate_log_config(config: Dict[str, Any]) -> None:
    """Validate logging configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if 'logging' not in config:
        raise ConfigurationError(
            "Missing 'logging' section in configuration",
            context={'error_code': ErrorCode.CONFIG_MISSING}
        )
        
    log_config = config['logging']
    if 'level' not in log_config:
        raise ConfigurationError(
            "Missing 'level' in logging configuration",
            context={'error_code': ErrorCode.CONFIG_MISSING}
        )
        
    if log_config['level'] not in VALID_LOG_LEVELS:
        raise ConfigurationError(
            f"Invalid log level: {log_config['level']}",
            context={
                'valid_levels': list(VALID_LOG_LEVELS),
                'error_code': ErrorCode.CONFIG_INVALID
            }
        )
        
    if 'format' not in log_config:
        raise ConfigurationError(
            "Missing 'format' in logging configuration",
            context={'error_code': ErrorCode.CONFIG_MISSING}
        )
        
    # Validate file size and backup count if specified
    if 'file_size' in log_config:
        try:
            file_size = int(log_config['file_size'])
            if file_size <= 0 or file_size > MAX_LOG_FILE_SIZE:
                raise ConfigurationError(
                    f"Invalid file size: {file_size}. Must be between 1 and {MAX_LOG_FILE_SIZE}",
                    context={'error_code': ErrorCode.CONFIG_INVALID}
                )
        except ValueError:
            raise ConfigurationError(
                "File size must be an integer",
                context={'error_code': ErrorCode.CONFIG_INVALID}
            )
            
    if 'backup_count' in log_config:
        try:
            backup_count = int(log_config['backup_count'])
            if backup_count < MIN_BACKUP_COUNT or backup_count > MAX_BACKUP_COUNT:
                raise ConfigurationError(
                    f"Invalid backup count: {backup_count}. Must be between {MIN_BACKUP_COUNT} and {MAX_BACKUP_COUNT}",
                    context={'error_code': ErrorCode.CONFIG_INVALID}
                )
        except ValueError:
            raise ConfigurationError(
                "Backup count must be an integer",
                context={'error_code': ErrorCode.CONFIG_INVALID}
            )

def ensure_log_directory(log_dir: str = DEFAULT_LOG_DIR) -> None:
    """Ensure log directory exists and is writable.
    
    Args:
        log_dir: Log directory path
        
    Raises:
        ConfigurationError: If directory cannot be created or accessed
    """
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Test write permissions
        test_file = os.path.join(log_dir, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except OSError as e:
        raise ConfigurationError(
            f"Failed to create or access log directory: {str(e)}",
            context={'error_code': ErrorCode.FILE_ACCESS_DENIED},
            cause=e
        )

def get_log_file_path(log_dir: str = DEFAULT_LOG_DIR) -> str:
    """Get log file path with timestamp.
    
    Args:
        log_dir: Log directory path
        
    Returns:
        str: Log file path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(log_dir, f"{DEFAULT_LOG_FILE}.{timestamp}")

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure logging based on settings.
    
    Args:
        config: Optional configuration dictionary. If not provided, uses default settings.
        
    Raises:
        ConfigurationError: If configuration is invalid or setup fails
    """
    if config is None:
        config = {
            'logging': {
                'level': LOG_LEVEL,
                'format': LOG_FORMAT,
                'file_size': MAX_LOG_FILE_SIZE,
                'backup_count': MAX_BACKUP_COUNT
            }
        }
    
    try:
        validate_log_config(config)
        ensure_log_directory()
        
        log_file = get_log_file_path()
        
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': config['logging']['format']
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
                },
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}'
                }
            },
            'handlers': {
                'console': {
                    'level': config['logging']['level'],
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'level': config['logging']['level'],
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file,
                    'maxBytes': config['logging'].get('file_size', MAX_LOG_FILE_SIZE),
                    'backupCount': config['logging'].get('backup_count', MAX_BACKUP_COUNT),
                    'encoding': 'utf-8'
                },
                'error_file': {
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': os.path.join(DEFAULT_LOG_DIR, 'error.log'),
                    'maxBytes': config['logging'].get('file_size', MAX_LOG_FILE_SIZE),
                    'backupCount': config['logging'].get('backup_count', MAX_BACKUP_COUNT),
                    'encoding': 'utf-8'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['console', 'file', 'error_file'],
                    'level': config['logging']['level'],
                    'propagate': True
                },
                'aws_rag': {  # Application logger
                    'handlers': ['console', 'file', 'error_file'],
                    'level': config['logging']['level'],
                    'propagate': False
                }
            }
        })
    except Exception as e:
        raise ConfigurationError(
            f"Failed to setup logging: {str(e)}",
            context={'error_code': ErrorCode.CONFIG_INVALID},
            cause=e
        )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        ConfigurationError: If logger name is invalid
    """
    if not name or not isinstance(name, str):
        raise ConfigurationError(
            "Invalid logger name",
            context={'error_code': ErrorCode.CONFIG_INVALID, 'name': name}
        )
    return logging.getLogger(f'aws_rag.{name}')

def get_error_logger() -> logging.Logger:
    """Get a logger instance specifically for error logging.
    
    Returns:
        logging.Logger: Configured error logger instance
    """
    return logging.getLogger('aws_rag.error') 