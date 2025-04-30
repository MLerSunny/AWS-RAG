from typing import List, Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class Chunker:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
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
                "end": end
            })
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """Process a document and return chunks with metadata."""
        try:
            text = document.get("text", "")
            source = document.get("source", "unknown")
            
            chunks = self.chunk_text(text)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    "source": source,
                    "document_id": document.get("id"),
                    "metadata": document.get("metadata", {})
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise 