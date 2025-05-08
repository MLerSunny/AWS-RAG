"""
Document chunking service for breaking documents into smaller pieces.
"""
from typing import List, Dict, Any, Optional, Tuple
import uuid
import re
import nltk
from ..utils.logger import setup_logger

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = setup_logger(__name__)

class Chunker:
    """Service for chunking documents into smaller pieces for embedding and indexing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size (int): Target size of chunks in characters
            chunk_overlap (int): Overlap between consecutive chunks in characters
            separator (str): Preferred separator for chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        logger.info(f"Initialized chunker with size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text (str): The text to chunk
            metadata (Optional[Dict]): Metadata to associate with each chunk
            
        Returns:
            List[Dict]: List of chunks with content and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []
            
        # Initialize base metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Use sentence boundaries for more natural chunks
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If this sentence alone exceeds the chunk size, we need to split it
            if sentence_length > self.chunk_size:
                # Process any accumulated content first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk_dict(chunk_text, metadata))
                    
                    # Reset with overlap if needed
                    current_chunk, current_chunk_size = self._get_overlap_from_text(
                        " ".join(current_chunk)
                    )
                
                # Split the long sentence using a sliding window
                sentence_chunks = self._split_long_sentence(sentence)
                
                # Add all but the last sentence chunk directly
                for i, sent_chunk in enumerate(sentence_chunks[:-1]):
                    chunks.append(self._create_chunk_dict(sent_chunk, metadata))
                
                # Keep the last part to continue with the next sentences
                current_chunk = [sentence_chunks[-1]]
                current_chunk_size = len(sentence_chunks[-1])
                
            # If adding this sentence would exceed the chunk size, finalize the current chunk
            elif current_chunk_size + sentence_length + 1 > self.chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk_dict(chunk_text, metadata))
                
                # New chunk with overlap
                current_chunk, current_chunk_size = self._get_overlap_from_text(chunk_text)
                
                # Add the current sentence to the new chunk
                current_chunk.append(sentence)
                current_chunk_size += sentence_length + 1  # +1 for space
                
            # Otherwise, add the sentence to the current chunk
            else:
                current_chunk.append(sentence)
                current_chunk_size += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk_dict(chunk_text, metadata))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a long sentence into smaller chunks.
        
        Args:
            sentence (str): The sentence to split
            
        Returns:
            List[str]: List of sentence chunks
        """
        # Use a sliding window with overlap
        chunks = []
        start = 0
        
        while start < len(sentence):
            # Find a good breakpoint
            end = min(start + self.chunk_size, len(sentence))
            
            # If we're not at the end, try to find a good break point
            if end < len(sentence):
                # Try to find a punctuation or space to break at
                for i in range(min(50, end - start)):
                    if sentence[end - i] in '.,;:!? ':
                        end = end - i + 1  # Include the punctuation
                        break
            
            chunks.append(sentence[start:end].strip())
            start = max(0, end - self.chunk_overlap)
        
        return chunks
    
    def _get_overlap_from_text(self, text: str) -> Tuple[List[str], int]:
        """
        Extract the overlap from the previous chunk to maintain context.
        
        Args:
            text (str): Previous chunk text
            
        Returns:
            Tuple[List[str], int]: List of sentences in overlap and total size
        """
        # No overlap needed if text is shorter than the overlap
        if len(text) <= self.chunk_overlap:
            return [text], len(text)
            
        # Get the last part of the text
        overlap_text = text[-self.chunk_overlap:]
        
        # Find a good starting point (start of a sentence if possible)
        match = re.search(r'[.!?]\s+\w', overlap_text)
        if match:
            # Start from the character after the punctuation and space
            pos = match.end() - 1
            overlap_text = overlap_text[pos:]
        
        return [overlap_text], len(overlap_text)
    
    def _create_chunk_dict(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a chunk dictionary with content and metadata.
        
        Args:
            content (str): The chunk content
            metadata (Dict): The metadata to include
            
        Returns:
            Dict: Chunk dictionary
        """
        # Create a copy of the metadata to avoid modifying the original
        chunk_metadata = metadata.copy()
        
        # Add chunk ID if not present
        if 'chunk_id' not in chunk_metadata:
            chunk_metadata['chunk_id'] = str(uuid.uuid4())
            
        return {
            'content': content,
            'metadata': chunk_metadata
        }
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document dictionary containing text and metadata.
        
        Args:
            document (Dict): Document with 'content' and 'metadata' fields
            chunk_size (Optional[int]): Override default chunk size
            chunk_overlap (Optional[int]): Override default chunk overlap
            
        Returns:
            List[Dict]: List of chunks
        """
        # Use provided values or defaults
        original_chunk_size = self.chunk_size
        original_chunk_overlap = self.chunk_overlap
        
        if chunk_size is not None:
            self.chunk_size = chunk_size
            
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        
        try:
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            # Add document_id to metadata if not present
            if 'document_id' not in metadata and 'id' in document:
                metadata['document_id'] = document['id']
                
            # Chunk the text
            chunks = self.chunk_text(content, metadata)
            
            return chunks
        finally:
            # Restore original values
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_chunk_overlap 