from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, ConnectionTimeout
from typing import List, Dict, Any, Optional
from ..config import settings
from ..utils.logger import setup_logger
import time

logger = setup_logger(__name__)

class OpenSearchClient:
    def __init__(self):
        self.host = settings.OPENSEARCH_HOST
        self.port = settings.OPENSEARCH_PORT
        self.index_name = settings.OPENSEARCH_INDEX
        self.connection_retry_count = 3
        self.connection_retry_delay = 2  # seconds
        self.client = None
        self._connect()
        self._ensure_index()
    
    def _connect(self) -> None:
        """Establish connection to OpenSearch with retry mechanism."""
        retry_count = 0
        
        while retry_count < self.connection_retry_count:
            try:
                self.client = OpenSearch(
                    hosts=[{
                        'host': self.host,
                        'port': self.port
                    }],
                    http_compress=True,
                    use_ssl=False,
                    verify_certs=False,
                    connection_class=RequestsHttpConnection,
                    timeout=30
                )
                
                # Test connection
                self.client.info()
                logger.info(f"Successfully connected to OpenSearch at {self.host}:{self.port}")
                return
                
            except (ConnectionError, ConnectionTimeout) as e:
                retry_count += 1
                logger.warning(f"Connection to OpenSearch failed (attempt {retry_count}/{self.connection_retry_count}): {str(e)}")
                
                if retry_count < self.connection_retry_count:
                    time.sleep(self.connection_retry_delay)
                else:
                    logger.error("Failed to connect to OpenSearch after multiple attempts")
                    # We'll continue and handle failures during operations
        
    def _ensure_index(self) -> None:
        """Create the index if it doesn't exist."""
        try:
            if self.client and not self.client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 1536  # Titan embedding dimension
                            },
                            "source": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created index '{self.index_name}' in OpenSearch")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            # Continue without throwing - we'll handle failures during indexing
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """Index a document with its embedding. Returns success status."""
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return False
            
        try:
            self.client.index(
                index=self.index_name,
                body=document,
                refresh=True
            )
            return True
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False
    
    def index_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Index multiple documents. Returns count of successfully indexed docs."""
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return 0
            
        success_count = 0
        for doc in documents:
            if self.index_document(doc):
                success_count += 1
        
        return success_count
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using k-NN."""
        if not self.client:
            logger.error("OpenSearch client not initialized")
            return []
            
        try:
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
            
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            return [
                {
                    "text": hit["_source"]["text"],
                    "source": hit["_source"].get("source", "unknown"),
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {})
                }
                for hit in response["hits"]["hits"]
            ]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return [] 