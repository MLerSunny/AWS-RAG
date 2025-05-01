"""
Configuration settings for the GenAI document ingestion service.
"""
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
import os
import json
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_secret_value(secret_name):
    """Retrieve a secret from AWS Secrets Manager."""
    if not secret_name:
        return None
        
    client = boto3.client(
        service_name='secretsmanager',
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            return json.loads(response['SecretString'])
        return None
    except Exception as e:
        print(f"Error fetching secret {secret_name}: {str(e)}")
        return None

# Get OpenSearch credentials from Secrets Manager
opensearch_secret_name = os.environ.get('OPENSEARCH_CREDENTIALS_SECRET')
opensearch_creds = get_secret_value(opensearch_secret_name)

# Get Bedrock credentials from Secrets Manager
bedrock_secret_name = os.environ.get('BEDROCK_CREDENTIALS_SECRET')
bedrock_creds = get_secret_value(bedrock_secret_name)

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # General settings
    APP_NAME: str = "GenAI Document Ingestion"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # AWS settings
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # OpenSearch settings
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_INDEX: str = "documents"
    OPENSEARCH_USERNAME: Optional[str] = opensearch_creds.get('username') if opensearch_creds else None
    OPENSEARCH_PASSWORD: Optional[str] = opensearch_creds.get('password') if opensearch_creds else None
    OPENSEARCH_AWS_AUTH: bool = False
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 20
    SUPPORTED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "md", "csv", "json"]
    
    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local model to use as fallback
    EMBEDDING_DIMENSION: int = 768
    USE_BEDROCK_EMBEDDINGS: bool = True  # Whether to use AWS Bedrock for embeddings
    
    # LLM settings
    BEDROCK_MODEL_ID: str = bedrock_creds.get('model_id') if bedrock_creds else "anthropic.claude-v2"
    TITAN_MODEL_ID: str = "amazon.titan-text-express-v1"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000
    
    # Storage settings
    STORAGE_DIR: str = "storage"
    FEEDBACK_FILE: str = "storage/feedback.jsonl"
    MODEL_REGISTRY_FILE: str = "storage/models.json"
    EXPERIMENTS_FILE: str = "storage/experiments.json"
    
    # SharePoint settings
    SHAREPOINT_SITE_URL: Optional[str] = None
    SHAREPOINT_USERNAME: Optional[str] = None
    SHAREPOINT_PASSWORD: Optional[str] = None
    
    # ServiceNow settings
    SERVICENOW_INSTANCE_URL: Optional[str] = None
    SERVICENOW_USERNAME: Optional[str] = None
    SERVICENOW_PASSWORD: Optional[str] = None
    
    # S3 settings for uploads/downloads
    S3_BUCKET: Optional[str] = None
    S3_PREFIX: str = "documents"
    
    # Fine-tuning settings
    FINETUNE_OUTPUT_BUCKET: Optional[str] = None
    FINETUNE_OUTPUT_PREFIX: str = "finetune-output"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Override settings with env vars
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure storage directory exists
os.makedirs(os.path.dirname(settings.FEEDBACK_FILE), exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_REGISTRY_FILE), exist_ok=True)
os.makedirs(os.path.dirname(settings.EXPERIMENTS_FILE), exist_ok=True) 