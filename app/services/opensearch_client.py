"""
OpenSearch client for vector storage and retrieval.
"""
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, Sequence, cast, Callable, Type
import json
import time
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, exceptions
import boto3
import numpy as np
from ..config import settings
from ..utils.logger import setup_logger
from ..utils.validation import validate_not_empty, validate_dict, validate_type, validate_range
import random
import uuid

logger = setup_logger(__name__)

class OpenSearchError(Exception):
    """Base exception for OpenSearch client errors."""
    def __init__(self, message: str, error_code: str = "OPENSEARCH_ERROR", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class ConnectionError(OpenSearchError):
    """Raised when connection to OpenSearch fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONNECTION_ERROR", context)

class IndexError(OpenSearchError):
    """Raised when index operation fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INDEX_ERROR", context)

class SearchError(OpenSearchError):
    """Raised when search operation fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SEARCH_ERROR", context)

class BulkOperationError(OpenSearchError):
    """Raised when bulk operation fails."""
    def __init__(self, message: str, failed_items: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "BULK_OPERATION_ERROR", context)
        self.failed_items = failed_items

T = TypeVar('T')

class OpenSearchClient:
    """Client for interacting with OpenSearch for vector search."""
    
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1  # seconds
    MAX_RETRY_DELAY = 10  # seconds
    BATCH_SIZE = 100  # Maximum number of documents per bulk operation
    MAX_DIMENSIONS = 4096  # Maximum vector dimensions supported by OpenSearch
    DEFAULT_KNN_ALGO = "hnsw"  # Default KNN algorithm
    DEFAULT_KNN_PARAMS = {
        "m": 16,  # Number of connections per node
        "ef_construction": 100,  # Size of the dynamic candidate list
        "ef_search": 100  # Size of the dynamic candidate list for search
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_ssl: bool = True,
        verify_certs: bool = True,
        region: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        """
        Initialize the OpenSearch client.
        
        Args:
            host (Optional[str]): OpenSearch host
            port (Optional[int]): OpenSearch port
            use_ssl (bool): Whether to use SSL
            verify_certs (bool): Whether to verify certificates
            region (Optional[str]): AWS region for authentication
            index_name (Optional[str]): Default index name
            
        Raises:
            ConnectionError: If connection to OpenSearch fails
        """
        # Configure from settings
        self.host = host or settings.OPENSEARCH_HOST
        self.port = port or settings.OPENSEARCH_PORT
        self.region = region or settings.AWS_REGION
        self.index_name = index_name or settings.OPENSEARCH_INDEX
        
        validate_not_empty(self.host, "host")
        validate_type(self.port, int, "port")
        validate_range(self.port, "port", 1, 65535)
        
        try:
            # Initialize OpenSearch client
            if settings.OPENSEARCH_AWS_AUTH:
                # Use AWS authentication
                credentials = boto3.Session().get_credentials()
                auth = AWSV4SignerAuth(credentials, self.region)
                
                self.client = OpenSearch(
                    hosts=[{'host': self.host, 'port': self.port}],
                    http_auth=auth,
                    use_ssl=use_ssl,
                    verify_certs=verify_certs,
                    connection_class=RequestsHttpConnection,
                    retry_on_timeout=True,
                    max_retries=self.MAX_RETRIES
                )
            else:
                # Use basic authentication if configured
                if settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
                    self.client = OpenSearch(
                        hosts=[{'host': self.host, 'port': self.port}],
                        http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
                        use_ssl=use_ssl,
                        verify_certs=verify_certs,
                        retry_on_timeout=True,
                        max_retries=self.MAX_RETRIES
                    )
                else:
                    # No authentication
                    self.client = OpenSearch(
                        hosts=[{'host': self.host, 'port': self.port}],
                        use_ssl=use_ssl,
                        verify_certs=verify_certs,
                        retry_on_timeout=True,
                        max_retries=self.MAX_RETRIES
                    )
                
            logger.info(f"Connected to OpenSearch at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {str(e)}")
            raise ConnectionError(f"Failed to connect to OpenSearch: {str(e)}")
    
    def _retry_operation(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Retry an operation with exponential backoff and jitter.
        
        Args:
            operation (Callable[..., Any]): Operation to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Any: Result of the operation
            
        Raises:
            OpenSearchError: If operation fails after retries
        """
        last_error = None
        delay = self.INITIAL_RETRY_DELAY
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except (exceptions.ConnectionError, exceptions.ConnectionTimeout) as e:
                last_error = e
                if attempt == self.MAX_RETRIES - 1:
                    break
                    
                # Add jitter to delay
                jitter = (0.5 + random.random()) * delay
                delay = min(delay * 2, self.MAX_RETRY_DELAY)
                
                logger.warning(f"Operation failed, retrying in {jitter:.2f}s: {str(e)}")
                time.sleep(jitter)
            except Exception as e:
                raise OpenSearchError(f"Operation failed: {str(e)}")
        
        raise OpenSearchError(f"Operation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    def is_connected(self) -> bool:
        """
        Check if the client is connected to OpenSearch.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.client:
            return False
            
        try:
            info = self._retry_operation(self.client.info)
            return 'version' in info
        except Exception:
            return False
    
    def _get_index_mapping(self, dimensions: int, knn_algo: Optional[str] = None, knn_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get index mapping with vector search capabilities.
        
        Args:
            dimensions (int): Dimensions of the embedding vectors
            knn_algo (Optional[str]): KNN algorithm to use
            knn_params (Optional[Dict]): KNN algorithm parameters
            
        Returns:
            Dict[str, Any]: Index mapping
        """
        knn_algo = knn_algo or self.DEFAULT_KNN_ALGO
        knn_params = knn_params or self.DEFAULT_KNN_PARAMS
        
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": knn_params["ef_search"],
                    "knn.algo_param.ef_construction": knn_params["ef_construction"],
                    "knn.algo_param.m": knn_params["m"]
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimensions,
                        "method": {
                            "name": knn_algo,
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": knn_params
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "tags": {"type": "keyword"},
                            "language": {"type": "keyword"},
                            "version": {"type": "keyword"}
                        }
                    }
                }
            }
        }
    
    def create_index(self, index_name: Optional[str] = None, dimensions: int = 768, knn_algo: Optional[str] = None, knn_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create an index with vector search capabilities if it doesn't exist.
        
        Args:
            index_name (Optional[str]): Name of the index
            dimensions (int): Dimensions of the embedding vectors
            knn_algo (Optional[str]): KNN algorithm to use
            knn_params (Optional[Dict]): KNN algorithm parameters
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            IndexError: If index creation fails
        """
        index_name = index_name or self.index_name
        validate_not_empty(index_name, "index_name")
        validate_range(dimensions, "dimensions", 1, self.MAX_DIMENSIONS)
        
        if not self.client:
            raise ConnectionError("OpenSearch client not initialized")
            
        try:
            # Check if index already exists
            if self._retry_operation(self.client.indices.exists, index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True
                
            # Get index mapping
            index_body = self._get_index_mapping(dimensions, knn_algo, knn_params)
            
            # Create the index
            self._retry_operation(self.client.indices.create, index=index_name, body=index_body)
            logger.info(f"Created index {index_name} with {dimensions} dimensions")
            return True
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {str(e)}")
            raise IndexError(f"Failed to create index {index_name}: {str(e)}")
    
    def _prepare_bulk_operation(self, documents: List[Dict[str, Any]], index_name: str) -> List[Dict[str, Any]]:
        """
        Prepare documents for bulk indexing.
        
        Args:
            documents (List[Dict]): Documents to index
            index_name (str): Target index name
            
        Returns:
            List[Dict]: Prepared bulk operation
        """
        bulk_data = []
        for doc in documents:
            # Add index action
            bulk_data.append({
                "index": {
                    "_index": index_name,
                    "_id": doc.get("document_id", str(uuid.uuid4()))
                }
            })
            
            # Add document data
            bulk_data.append({
                "embedding": doc["embedding"],
                "content": doc["content"],
                "metadata": {
                    **doc.get("metadata", {}),
                    "updated_at": int(time.time())
                }
            })
        
        return bulk_data
    
    def bulk_index(
        self, 
        documents: List[Dict[str, Any]], 
        index_name: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Bulk index documents with batching.
        
        Args:
            documents (List[Dict]): Documents to index
            index_name (Optional[str]): Target index name
            batch_size (Optional[int]): Batch size for bulk operations
            
        Returns:
            Dict[str, int]: Statistics about the operation
            
        Raises:
            BulkOperationError: If bulk operation fails
        """
        index_name = index_name or self.index_name
        batch_size = batch_size or self.BATCH_SIZE
        
        if not self.client:
            raise ConnectionError("OpenSearch client not initialized")
            
        try:
            total_docs = len(documents)
            successful = 0
            failed = []
            
            # Process in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                bulk_data = self._prepare_bulk_operation(batch, index_name)
                
                try:
                    response = self._retry_operation(
                        self.client.bulk,
                        body=bulk_data,
                        refresh=True
                    )
                    
                    # Check for errors
                    if response.get("errors"):
                        for item in response["items"]:
                            if "index" in item and item["index"].get("error"):
                                failed.append({
                                    "document_id": item["index"]["_id"],
                                    "error": item["index"]["error"]
                                })
                            else:
                                successful += 1
                    else:
                        successful += len(batch)
                        
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    failed.extend([{
                        "document_id": doc.get("document_id", "unknown"),
                        "error": str(e)
                    } for doc in batch])
            
            if failed:
                raise BulkOperationError(
                    f"Bulk indexing partially failed: {len(failed)}/{total_docs} documents failed",
                    failed
                )
            
            return {
                "total": total_docs,
                "successful": successful,
                "failed": len(failed)
            }
            
        except Exception as e:
            if isinstance(e, BulkOperationError):
                raise
            logger.error(f"Error in bulk indexing: {str(e)}")
            raise BulkOperationError(f"Bulk indexing failed: {str(e)}", [])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        index_name: Optional[str] = None,
        filter_query: Optional[Dict] = None,
        min_score: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding (List[float]): Query vector
            top_k (int): Number of results to return
            index_name (Optional[str]): Target index name
            filter_query (Optional[Dict]): Filter query to apply
            min_score (Optional[float]): Minimum similarity score
            include_metadata (bool): Whether to include metadata in results
            
        Returns:
            List[Dict]: Search results
            
        Raises:
            SearchError: If search operation fails
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            raise ConnectionError("OpenSearch client not initialized")
            
        try:
            # Build search query
            query = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "embedding",
                                "query_value": query_embedding,
                                "space_type": "l2"
                            }
                        }
                    }
                }
            }
            
            # Add filter if provided
            if filter_query:
                query["query"] = {
                    "bool": {
                        "must": [
                            query["query"],
                            filter_query
                        ]
                    }
                }
            
            # Add minimum score if provided
            if min_score is not None:
                query["min_score"] = min_score
            
            # Add source filtering
            if not include_metadata:
                query["_source"] = ["content"]
            
            # Execute search
            response = self._retry_operation(
                self.client.search,
                index=index_name,
                body=query
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "text": hit["_source"]["content"]
                }
                
                if include_metadata and "metadata" in hit["_source"]:
                    result["metadata"] = hit["_source"]["metadata"]
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise SearchError(f"Failed to search documents: {str(e)}")
    
    def delete_document(
        self, 
        document_id: str, 
        index_name: Optional[str] = None
    ) -> bool:
        """
        Delete a document from the index.
        
        Args:
            document_id (str): ID of the document to delete
            index_name (Optional[str]): Target index name
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            IndexError: If delete operation fails
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            raise ConnectionError("OpenSearch client not initialized")
            
        try:
            response = self._retry_operation(
                self.client.delete,
                index=index_name,
                id=document_id
            )
            
            return response["result"] == "deleted"
            
        except exceptions.NotFoundError:
            logger.warning(f"Document {document_id} not found in index {index_name}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise IndexError(f"Failed to delete document {document_id}: {str(e)}") 