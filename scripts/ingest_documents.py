import os
import json
import argparse
from typing import List, Dict
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.opensearch_client import OpenSearchClient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def process_document(file_path: str) -> Dict:
    """Process a single document file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return {
            "text": text,
            "source": file_path,
            "metadata": {
                "filename": os.path.basename(file_path),
                "size": os.path.getsize(file_path)
            }
        }
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("input_dir", help="Directory containing documents to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    args = parser.parse_args()
    
    # Initialize services
    chunker = Chunker()
    embedder = Embedder()
    opensearch = OpenSearchClient()
    
    # Process documents
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(root, file)
                logger.info(f"Processing {file_path}")
                
                # Process document
                document = process_document(file_path)
                if document:
                    # Generate chunks
                    chunks = chunker.chunk_document(document)
                    
                    # Generate embeddings and index
                    for chunk in chunks:
                        embedding = embedder.embed_text(chunk['text'])
                        chunk['embedding'] = embedding
                        opensearch.index_document(chunk)
                    
                    logger.info(f"Successfully processed {file_path}")

if __name__ == "__main__":
    main() 