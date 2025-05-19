from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from ..services.bedrock_llm import BedrockLLM
from ..services.opensearch_client import OpenSearchClient
from ..services.embedder import Embedder
from ..utils.logger import setup_logger
from ..utils.validation import validate_query
from ..utils.cache import Cache
import time

router = APIRouter()
logger = setup_logger(__name__)

# Initialize cache
cache = Cache()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_local_embeddings: bool = Field(default=True)
    
    @validator('query')
    def validate_query_text(cls, v):
        if not validate_query(v):
            raise ValueError("Query contains invalid characters or is too short")
        return v

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    success: bool = True
    error: Optional[str] = None
    cache_hit: bool = False
    processing_time: float

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    req: Request
):
    start_time = time.time()
    
    # Check cache first
    cache_key = f"query:{request.query}:{request.top_k}:{request.use_local_embeddings}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return QueryResponse(
            **cached_result,
            cache_hit=True,
            processing_time=time.time() - start_time
        )
    
    try:
        # Initialize services
        embedder = Embedder()
        opensearch = OpenSearchClient()
        llm = BedrockLLM()
        
        # Get query embedding
        try:
            query_embedding = embedder.embed_text(
                request.query, 
                use_titan=not request.use_local_embeddings
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embeddings: {str(e)}"
            )
        
        # Search for relevant documents
        try:
            results = opensearch.search(
                query_embedding=query_embedding,
                top_k=request.top_k
            )
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error searching documents: {str(e)}"
            )
        
        if not results:
            response = QueryResponse(
                answer="I don't have any relevant information to answer your question. Please try a different query or check if documents have been ingested.",
                sources=[],
                success=True,
                processing_time=time.time() - start_time
            )
            cache.set(cache_key, response.dict())
            return response
        
        # Generate response using LLM
        context = "\n".join([doc["text"] for doc in results])
        
        try:
            answer = llm.generate_response(
                query=request.query,
                context=context
            )
        except Exception as e:
            logger.error(f"Error generating response with LLM: {str(e)}")
            # Fallback to just returning the most relevant context
            response = QueryResponse(
                answer=f"I found relevant information but encountered an error generating a response. Here's the most relevant information I found:\n\n{results[0]['text']}",
                sources=[doc["source"] for doc in results],
                success=False,
                error=f"LLM error: {str(e)}",
                processing_time=time.time() - start_time
            )
            cache.set(cache_key, response.dict())
            return response
        
        response = QueryResponse(
            answer=answer,
            sources=[doc["source"] for doc in results],
            success=True,
            processing_time=time.time() - start_time
        )
        
        # Cache successful responses
        cache.set(cache_key, response.dict())
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 