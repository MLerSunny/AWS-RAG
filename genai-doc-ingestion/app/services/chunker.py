from typing import List, Dict, Any
from ..config import settings
from ..utils.logger import setup_logger
import uuid

logger = setup_logger(__name__)

class Chunker:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "start": start,
                "end": end,
                "id": str(uuid.uuid4())
            })
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            if start >= len(text) or start == end:
                break
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a document and return chunks with metadata."""
        try:
            text = document.get("text", "")
            source = document.get("source", "unknown")
            doc_id = document.get("id", str(uuid.uuid4()))
            
            if not text:
                logger.warning(f"Empty text content for document: {source}")
                return []
                
            chunks = self.chunk_text(text)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    "source": source,
                    "document_id": doc_id,
                    "metadata": document.get("metadata", {})
                })
            
            logger.info(f"Document '{source}' chunked into {len(chunks)} parts")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise 