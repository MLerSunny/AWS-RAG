"""
GenAI Document Ingestion API Startup Script

This script handles environment initialization and starts the API server.
"""
import os
import argparse
import sys
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import setup_logger

# Ensure app module is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = setup_logger(__name__)

def create_directories():
    """Create necessary directories for the application."""
    directories = ["logs", "uploads", "temp"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def load_environment(env_file=".env"):
    """Load environment variables from .env file."""
    # Check if .env exists, create a default one if not
    if not os.path.exists(env_file):
        logger.warning(f"{env_file} not found, creating default configuration")
        with open(env_file, "w") as f:
            f.write("""# AWS Configuration
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
""")
    
    # Load environment variables
    load_dotenv(env_file)
    logger.info(f"Loaded environment from {env_file}")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="GenAI Document Ingestion API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Initialize environment
    load_environment(args.env)
    create_directories()
    
    # Start the server
    logger.info(f"Starting server at {args.host}:{args.port} (reload={args.reload})")
    uvicorn.run(
        "app.main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )

if __name__ == "__main__":
    main() 