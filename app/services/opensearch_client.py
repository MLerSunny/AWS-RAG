"""
OpenSearch client for vector storage and retrieval.
"""
from typing import Dict, List, Optional, Any, Union
import json
import time
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
import numpy as np
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenSearchClient:
    """Client for interacting with OpenSearch for vector search."""
    
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
        """
        # Configure from settings
        self.host = host or settings.OPENSEARCH_HOST
        self.port = port or settings.OPENSEARCH_PORT
        self.region = region or settings.AWS_REGION
        self.index_name = index_name or settings.OPENSEARCH_INDEX
        
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
                    connection_class=RequestsHttpConnection
                )
            else:
                # Use basic authentication if configured
                if settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
                    self.client = OpenSearch(
                        hosts=[{'host': self.host, 'port': self.port}],
                        http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
                        use_ssl=use_ssl,
                        verify_certs=verify_certs
                    )
                else:
                    # No authentication
                    self.client = OpenSearch(
                        hosts=[{'host': self.host, 'port': self.port}],
                        use_ssl=use_ssl,
                        verify_certs=verify_certs
                    )
                
            logger.info(f"Connected to OpenSearch at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """
        Check if the client is connected to OpenSearch.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.client:
            return False
            
        try:
            info = self.client.info()
            return 'version' in info
        except Exception:
            return False
    
    def create_index(self, index_name: Optional[str] = None, dimensions: int = 768) -> bool:
        """
        Create an index with vector search capabilities if it doesn't exist.
        
        Args:
            index_name (Optional[str]): Name of the index
            dimensions (int): Dimensions of the embedding vectors
            
        Returns:
            bool: True if successful, False otherwise
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return False
            
        try:
            # Check if index already exists
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True
                
            # Define index mapping with vector field
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimensions
                        },
                        "content": {
                            "type": "text"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "keyword"},
                                "source": {"type": "keyword"},
                                "chunk_id": {"type": "keyword"},
                                "document_id": {"type": "keyword"},
                                "created_at": {"type": "date"},
                                "updated_at": {"type": "date"}
                            }
                        }
                    }
                }
            }
            
            # Create the index
            self.client.indices.create(index=index_name, body=index_body)
            logger.info(f"Created index {index_name} with {dimensions} dimensions")
            return True
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {str(e)}")
            return False
    
    def index_document(
        self,
        document_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any],
        index_name: Optional[str] = None
    ) -> bool:
        """
        Index a document with its embedding vector.
        
        Args:
            document_id (str): Unique ID for the document
            embedding (List[float]): Vector embedding for the document
            content (str): Text content of the document
            metadata (Dict): Additional metadata
            index_name (Optional[str]): Index name
            
        Returns:
            bool: True if successful, False otherwise
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return False
            
        try:
            # Add timestamps if not already present
            if 'created_at' not in metadata:
                metadata['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            metadata['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            # Prepare document
            document = {
                'embedding': embedding,
                'content': content,
                'metadata': metadata
            }
            
            # Index the document
            self.client.index(
                index=index_name,
                body=document,
                id=document_id,
                refresh=True
            )
            
            logger.info(f"Indexed document {document_id} in {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {str(e)}")
            return False
    
    def bulk_index(
        self, 
        documents: List[Dict[str, Any]], 
        index_name: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Index multiple documents in bulk.
        
        Args:
            documents (List[Dict]): List of documents with id, embedding, content, metadata
            index_name (Optional[str]): Index name
            
        Returns:
            Dict[str, int]: Result with success and failure counts
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return {"successful": 0, "failed": len(documents)}
            
        try:
            if not documents:
                return {"successful": 0, "failed": 0}
                
            # Prepare bulk request body
            bulk_body = []
            
            # Add each document to the bulk request
            for doc in documents:
                # Add timestamps if not present
                if 'metadata' not in doc:
                    doc['metadata'] = {}
                    
                if 'created_at' not in doc['metadata']:
                    doc['metadata']['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                
                doc['metadata']['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                
                # Add index operation
                bulk_body.append({"index": {"_index": index_name, "_id": doc['id']}})
                
                # Add document (without the id, as it's specified in the operation)
                document = {
                    'embedding': doc['embedding'],
                    'content': doc['content'],
                    'metadata': doc['metadata']
                }
                bulk_body.append(document)
            
            # Execute bulk operation
            response = self.client.bulk(body=bulk_body, refresh=True)
            
            # Count successes and failures
            successful = len([item for item in response['items'] if item.get('index', {}).get('status') < 300])
            failed = len(documents) - successful
            
            logger.info(f"Bulk indexed {successful} documents, {failed} failed")
            return {"successful": successful, "failed": failed}
        except Exception as e:
            logger.error(f"Error in bulk indexing: {str(e)}")
            return {"successful": 0, "failed": len(documents)}
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        index_name: Optional[str] = None,
        filter_query: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding (List[float]): The query vector
            top_k (int): Number of results to return
            index_name (Optional[str]): Index name
            filter_query (Optional[Dict]): Additional query filters
            
        Returns:
            List[Dict]: Search results
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return []
            
        try:
            # Construct the search query
            query = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                }
            }
            
            # Add filter if provided
            if filter_query:
                query["query"] = {
                    "bool": {
                        "must": [query["query"]],
                        "filter": filter_query
                    }
                }
            
            # Execute search
            response = self.client.search(
                body=query,
                index=index_name
            )
            
            # Format results
            hits = response["hits"]["hits"]
            logger.info(f"Found {len(hits)} results for vector search")
            
            return hits
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def delete_document(
        self, 
        document_id: str, 
        index_name: Optional[str] = None
    ) -> bool:
        """
        Delete a document from the index.
        
        Args:
            document_id (str): Document ID to delete
            index_name (Optional[str]): Index name
            
        Returns:
            bool: True if successful, False otherwise
        """
        index_name = index_name or self.index_name
        
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return False
            
        try:
            response = self.client.delete(
                index=index_name,
                id=document_id,
                refresh=True
            )
            
            result = response.get('result') == 'deleted'
            logger.info(f"Deleted document {document_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False 