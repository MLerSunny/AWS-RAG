from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import PyPDF2
from docx import Document
import os
import json
from ..services.chunker import Chunker
from ..services.embedder import Embedder
from ..services.opensearch_client import OpenSearchClient
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

class IngestResponse(BaseModel):
    message: str
    document_id: str
    chunks_processed: int
    success: bool

class BatchIngestResponse(BaseModel):
    message: str
    documents_processed: int
    total_chunks_processed: int
    success: bool
    failed_documents: List[str] = []

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text based on file type."""
    # Get file extension
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()
    
    # Process based on file type
    if file_extension == '.pdf':
        # Handle PDF
        with io.BytesIO(file_content) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
    elif file_extension == '.docx':
        # Handle DOCX
        with io.BytesIO(file_content) as docx_file:
            doc = Document(docx_file)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text
    elif file_extension == '.txt':
        # Handle TXT
        return file_content.decode('utf-8')
    else:
        # For unknown types, attempt to decode as text
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {file_extension}")

async def process_document_async(
    content: bytes, 
    filename: str, 
    document_id: Optional[str] = None, 
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process a document asynchronously and return results."""
    try:
        # Extract text based on file type
        content_str = extract_text_from_file(content, filename)
        
        # Initialize services
        chunker = Chunker()
        embedder = Embedder()
        opensearch = OpenSearchClient()
        
        # Create document dict expected by chunker
        document = {
            "text": content_str,
            "source": filename,
            "id": document_id or filename,
            "metadata": metadata or {}
        }
        
        # Process document
        chunks = chunker.chunk_document(document)
        if not chunks:
            return {
                "success": False,
                "document_id": document["id"],
                "chunks_processed": 0,
                "error": "No chunks were generated from the document"
            }
        
        # Generate embeddings and index chunks
        total_chunks = 0
        successful_chunks = 0
        doc_id = document["id"]
        
        for chunk in chunks:
            try:
                # Generate embedding for the chunk
                embedding = embedder.embed_text(chunk["text"])
                
                # Create document to index
                index_doc = {
                    "text": chunk["text"],
                    "embedding": embedding,
                    "source": chunk["source"],
                    "document_id": doc_id,
                    "metadata": {
                        "start": chunk["start"],
                        "end": chunk["end"],
                        **chunk.get("metadata", {})
                    }
                }
                
                # Index the document
                if opensearch.index_document(index_doc):
                    successful_chunks += 1
                
                total_chunks += 1
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
        
        return {
            "success": successful_chunks > 0,
            "document_id": doc_id,
            "chunks_processed": successful_chunks,
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")
        return {
            "success": False,
            "document_id": document_id or filename,
            "chunks_processed": 0,
            "error": str(e)
        }

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    metadata: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    try:
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Read file content
        content = await file.read()
        
        # Process the document
        result = await process_document_async(
            content=content,
            filename=file.filename,
            document_id=document_id,
            metadata=meta_dict
        )
        
        if not result["success"]:
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to process document - no chunks were successfully indexed"
                )
        
        return IngestResponse(
            message=f"Successfully ingested document: {file.filename}",
            document_id=result["document_id"],
            chunks_processed=result["chunks_processed"],
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-ingest", response_model=BatchIngestResponse)
async def batch_ingest_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    try:
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        total_documents = 0
        total_chunks = 0
        failed_documents = []
        
        for file in files:
            try:
                # Read file content
                content = await file.read()
                
                # Process the document
                result = await process_document_async(
                    content=content,
                    filename=file.filename,
                    metadata=meta_dict
                )
                
                if result["success"]:
                    total_documents += 1
                    total_chunks += result["chunks_processed"]
                else:
                    failed_documents.append(file.filename)
                    
            except Exception as e:
                logger.warning(f"Error processing {file.filename}: {str(e)}")
                failed_documents.append(file.filename)
                continue
        
        # Determine if the operation was at least partially successful
        if total_documents == 0:
            message = "No documents were successfully processed"
            success = False
        else:
            success = True
            if failed_documents:
                message = f"Partially successful: {total_documents} documents processed, {len(failed_documents)} failed"
            else:
                message = f"Successfully ingested {total_documents} documents"
            
        return BatchIngestResponse(
            message=message,
            documents_processed=total_documents,
            total_chunks_processed=total_chunks,
            success=success,
            failed_documents=failed_documents
        )
        
    except Exception as e:
        logger.error(f"Error in batch ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 