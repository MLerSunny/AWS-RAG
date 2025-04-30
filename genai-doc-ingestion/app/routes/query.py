from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..services.bedrock_llm import BedrockLLM
from ..services.opensearch_client import OpenSearchClient
from ..services.embedder import Embedder
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_local_embeddings: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    success: bool = True
    error: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
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
            return QueryResponse(
                answer="I encountered an error generating embeddings for your query. Please try again later.",
                sources=[],
                success=False,
                error=f"Embedding error: {str(e)}"
            )
        
        # Search for relevant documents
        results = opensearch.search(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        if not results:
            return QueryResponse(
                answer="I don't have any relevant information to answer your question. Please try a different query or check if documents have been ingested.",
                sources=[],
                success=True
            )
        
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
            return QueryResponse(
                answer=f"I found relevant information but encountered an error generating a response. Here's the most relevant information I found:\n\n{results[0]['text']}",
                sources=[doc["source"] for doc in results],
                success=False,
                error=f"LLM error: {str(e)}"
            )
        
        return QueryResponse(
            answer=answer,
            sources=[doc["source"] for doc in results],
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer="I encountered an error processing your query. Please try again later.",
            sources=[],
            success=False,
            error=str(e)
        ) 