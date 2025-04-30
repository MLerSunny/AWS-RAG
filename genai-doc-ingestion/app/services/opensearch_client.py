from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import List, Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenSearchClient:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{
                'host': settings.OPENSEARCH_HOST,
                'port': settings.OPENSEARCH_PORT
            }],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
        self.index_name = settings.OPENSEARCH_INDEX
        self._ensure_index()
    
    def _ensure_index(self):
        """Create the index if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
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
    
    def index_document(self, document: Dict):
        """Index a document with its embedding."""
        try:
            self.client.index(
                index=self.index_name,
                body=document,
                refresh=True
            )
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    def index_batch(self, documents: List[Dict]):
        """Index multiple documents."""
        try:
            for doc in documents:
                self.index_document(doc)
        except Exception as e:
            logger.error(f"Error indexing batch: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents using k-NN."""
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
                    "source": hit["_source"]["source"],
                    "score": hit["_score"]
                }
                for hit in response["hits"]["hits"]
            ]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise 