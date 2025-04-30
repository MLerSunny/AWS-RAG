import json
import boto3
from typing import Dict, Any
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.opensearch_client import OpenSearchClient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for S3 document ingestion."""
    try:
        # Initialize services
        chunker = Chunker()
        embedder = Embedder()
        opensearch = OpenSearchClient()
        
        # Get S3 event details
        s3_event = event['Records'][0]['s3']
        bucket = s3_event['bucket']['name']
        key = s3_event['object']['key']
        
        # Get document from S3
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        document_text = response['Body'].read().decode('utf-8')
        
        # Create document object
        document = {
            "text": document_text,
            "source": f"s3://{bucket}/{key}",
            "metadata": {
                "bucket": bucket,
                "key": key,
                "size": response['ContentLength']
            }
        }
        
        # Process document
        chunks = chunker.chunk_document(document)
        
        # Generate embeddings and index
        for chunk in chunks:
            embedding = embedder.embed_text(chunk['text'])
            chunk['embedding'] = embedding
            opensearch.index_document(chunk)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Document processed successfully",
                "chunks_processed": len(chunks)
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        } 