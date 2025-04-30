import boto3
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Embedder:
    def __init__(self):
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text_titan(self, text: str) -> List[float]:
        """Generate embeddings using Amazon Titan."""
        try:
            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=text
            )
            embedding = response.get("embedding", [])
            return embedding
        except Exception as e:
            logger.error(f"Error generating Titan embedding: {str(e)}")
            raise
    
    def embed_text_local(self, text: str) -> List[float]:
        """Generate embeddings using local SentenceTransformer model."""
        try:
            embedding = self.local_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating local embedding: {str(e)}")
            raise
    
    def embed_text(self, text: str, use_titan: bool = True) -> List[float]:
        """Generate embeddings using either Titan or local model."""
        if use_titan and settings.AWS_ACCESS_KEY_ID:
            return self.embed_text_titan(text)
        return self.embed_text_local(text)
    
    def embed_batch(self, texts: List[str], use_titan: bool = True) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if use_titan and settings.AWS_ACCESS_KEY_ID:
            return [self.embed_text_titan(text) for text in texts]
        embeddings = self.local_model.encode(texts)
        return embeddings.tolist() 