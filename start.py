"""
GenAI Document Ingestion API Startup Script

This script handles environment initialization and starts the API server.
"""
import os
import argparse
import sys
import uvicorn
import atexit
import shutil
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from app.utils.logger import setup_logger
from exceptions import ConfigurationError, ErrorCode

# Ensure app module is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = setup_logger(__name__)

# Required environment variables
REQUIRED_ENV_VARS = {
    'AWS_REGION': str,
    'OPENSEARCH_HOST': str,
    'OPENSEARCH_PORT': int,
    'OPENSEARCH_INDEX': str,
    'BEDROCK_MODEL_ID': str,
    'ENVIRONMENT': str,
    'DEBUG': bool,
    'CHUNK_SIZE': int,
    'CHUNK_OVERLAP': int
}

def create_directories() -> None:
    """Create necessary directories for the application."""
    directories = ["logs", "uploads", "temp"]
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                raise ConfigurationError(
                    f"Failed to create directory {directory}",
                    context={'error_code': ErrorCode.FILE_ACCESS_DENIED},
                    cause=e
                )

def validate_environment() -> None:
    """Validate environment variables."""
    missing_vars = []
    invalid_vars = []
    
    for var_name, var_type in REQUIRED_ENV_VARS.items():
        value = os.getenv(var_name)
        if value is None:
            missing_vars.append(var_name)
            continue
            
        try:
            if var_type == bool:
                if value.lower() not in ('true', 'false', '1', '0'):
                    invalid_vars.append((var_name, value))
            elif var_type == int:
                int(value)
            # str type doesn't need validation
        except ValueError:
            invalid_vars.append((var_name, value))
    
    if missing_vars or invalid_vars:
        error_context = {
            'missing_vars': missing_vars,
            'invalid_vars': [f"{name}={value}" for name, value in invalid_vars]
        }
        raise ConfigurationError(
            "Invalid environment configuration",
            context={'error_code': ErrorCode.CONFIG_INVALID, **error_context}
        )

def create_default_env_file(env_file: str) -> None:
    """Create default .env file if it doesn't exist."""
    default_config = """# AWS Configuration
AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=documents

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-v2

# Application Settings
ENVIRONMENT=development
DEBUG=True
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""
    try:
        with open(env_file, "w") as f:
            f.write(default_config)
        logger.info(f"Created default {env_file}")
    except OSError as e:
        raise ConfigurationError(
            f"Failed to create {env_file}",
            context={'error_code': ErrorCode.FILE_WRITE_ERROR},
            cause=e
        )

def load_environment(env_file: str = ".env") -> None:
    """Load environment variables from .env file."""
    if not os.path.exists(env_file):
        logger.warning(f"{env_file} not found, creating default configuration")
        create_default_env_file(env_file)
    
    try:
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")
        validate_environment()
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load environment from {env_file}",
            context={'error_code': ErrorCode.CONFIG_INVALID},
            cause=e
        )

def cleanup() -> None:
    """Clean up temporary files and directories."""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="GenAI Document Ingestion API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    try:
        # Initialize environment
        load_environment(args.env)
        create_directories()
        
        # Register cleanup function
        atexit.register(cleanup)
        
        # Start the server
        logger.info(f"Starting server at {args.host}:{args.port} (reload={args.reload})")
        uvicorn.run(
            "app.main:app", 
            host=args.host, 
            port=args.port, 
            reload=args.reload
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 