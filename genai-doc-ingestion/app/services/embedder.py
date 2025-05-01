import boto3
import json
from typing import List, Dict, Union, Optional
import numpy as np
from functools import lru_cache
import hashlib
from sentence_transformers import SentenceTransformer
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Embedder:
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the embedding service.
        
        Args:
            cache_size (int): Number of embeddings to cache
        """
        # Initialize Bedrock client if credentials are available
        self.bedrock = None
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            self.bedrock = boto3.client(
                "bedrock-runtime",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        
        # Initialize local model with small but effective embedding model
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized local embedding model (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.error(f"Error loading local embedding model: {str(e)}")
            self.local_model = None
        
        self.cache_size = cache_size
        self._embedding_cache = {}  # Simple in-memory cache to complement LRU cache
        logger.info(f"Initialized embedder with cache size {cache_size}")
    
    @lru_cache(maxsize=1000)
    def _get_embedding_cached(self, text_hash: str, use_titan: bool) -> List[float]:
        """
        Internal cached embedding function based on text hash.
        
        Args:
            text_hash (str): Hash of the text to embed
            use_titan (bool): Whether to use the Titan model
            
        Returns:
            List[float]: The embedding vector
        """
        # Return the embedding from our internal cache if it exists
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
            
        # Indicate a cache miss - this should be handled by the caller
        # by generating a new embedding and updating the cache
        return []
    
    def _hash_text(self, text: str) -> str:
        """
        Create a hash of the text for caching.
        
        Args:
            text (str): Text to hash
            
        Returns:
            str: Hash of the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_text_titan(self, text: str) -> List[float]:
        """
        Generate embeddings using Amazon Titan.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.bedrock:
            raise ValueError("Bedrock client not initialized. Check AWS credentials.")
            
        try:
            # Prepare the request body
            body = json.dumps({"inputText": text})
            
            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=body
            )
            
            # Parse the response
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding", [])
            
            if not embedding:
                raise ValueError("Empty embedding returned from Titan")
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating Titan embedding: {str(e)}")
            # Fall back to local model if available
            if self.local_model:
                logger.info("Falling back to local embedding model")
                return self.embed_text_local(text)
            raise
    
    def embed_text_local(self, text: str) -> List[float]:
        """
        Generate embeddings using local SentenceTransformer model.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.local_model:
            raise ValueError("Local embedding model not initialized")
            
        try:
            embedding = self.local_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating local embedding: {str(e)}")
            raise
    
    def get_embeddings(self, text: str, use_titan: bool = True) -> List[float]:
        """
        Generate embeddings with caching.
        
        Args:
            text (str): Text to embed
            use_titan (bool): Whether to use the Titan model
            
        Returns:
            List[float]: Embedding vector
        """
        # Use caching for efficiency
        text_hash = self._hash_text(text)
        
        # Check if we have this in cache already
        cached_result = self._get_embedding_cached(text_hash, use_titan)
        if len(cached_result) > 0:  # If we got a valid result from cache
            return cached_result
        
        # Generate new embedding
        if use_titan and self.bedrock:
            result = self.embed_text_titan(text)
        else:
            result = self.embed_text_local(text)
        
        # Update cache
        self._embedding_cache[text_hash] = result
        # Clear and update LRU cache to ensure consistent behavior
        self._get_embedding_cached.cache_clear()
        self._get_embedding_cached(text_hash, use_titan)
        
        return result
    
    def embed_batch(self, texts: List[str], use_titan: bool = True, batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): Texts to embed
            use_titan (bool): Whether to use the Titan model
            batch_size (int): Batch size for local embedding
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        # For small batches, leverage caching with individual requests
        if len(texts) <= 5:
            return [self.get_embeddings(text, use_titan) for text in texts]
        
        # For Titan, we still need to make individual requests
        if use_titan and self.bedrock:
            logger.info(f"Embedding batch of {len(texts)} texts with Titan")
            results = []
            for i, text in enumerate(texts):
                try:
                    results.append(self.embed_text_titan(text))
                    if (i + 1) % 10 == 0:
                        logger.info(f"Embedded {i+1}/{len(texts)} texts")
                except Exception as e:
                    logger.error(f"Error embedding text {i}: {str(e)}")
                    # Fall back to local for this specific text
                    if self.local_model:
                        results.append(self.embed_text_local(text))
                    else:
                        # Use zeros as placeholder if everything fails
                        results.append([0.0] * 768)  # Standard embedding size
            return results
        
        # For local model, use efficient batching
        if not self.local_model:
            raise ValueError("Local embedding model not initialized")
            
        logger.info(f"Embedding batch of {len(texts)} texts with local model")
        
        # Process in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = self.local_model.encode(batch)
                all_embeddings.extend(embeddings.tolist())
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                # Use zeros as placeholder
                all_embeddings.extend([[0.0] * 768 for _ in batch])
        
        return all_embeddings
    
    def embed_text(self, text: str, use_titan: bool = True) -> List[float]:
        """
        Legacy method for backward compatibility.
        
        Args:
            text (str): Text to embed
            use_titan (bool): Whether to use the Titan model
            
        Returns:
            List[float]: Embedding vector
        """
        return self.get_embeddings(text, use_titan) 