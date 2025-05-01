from pydantic_settings import BaseSettings
from typing import Optional
import os
import json
import boto3

def get_secret_value(secret_name):
    """Retrieve a secret from AWS Secrets Manager."""
    if not secret_name:
        return None
        
    session = boto3.session.Session()
    client = session.client(
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
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # OpenSearch Configuration
    OPENSEARCH_HOST: str = os.environ.get('OPENSEARCH_HOST', 'localhost')
    OPENSEARCH_PORT: int = int(os.environ.get('OPENSEARCH_PORT', '9200'))
    OPENSEARCH_INDEX: str = "documents"
    OPENSEARCH_USERNAME: Optional[str] = opensearch_creds.get('username') if opensearch_creds else None
    OPENSEARCH_PASSWORD: Optional[str] = opensearch_creds.get('password') if opensearch_creds else None
    
    # Bedrock Configuration
    BEDROCK_MODEL_ID: str = bedrock_creds.get('model_id') if bedrock_creds else "anthropic.claude-v2"
    
    # Application Settings
    ENVIRONMENT: str = os.environ.get('ENVIRONMENT', 'development')
    DEBUG: bool = False
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings() 