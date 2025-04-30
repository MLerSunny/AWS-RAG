from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.bedrock_llm import BedrockLLM
from app.services.opensearch_client import OpenSearchClient
from app.services.embedder import Embedder
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Initialize services
        embedder = Embedder()
        opensearch = OpenSearchClient()
        llm = BedrockLLM()
        
        # Get query embedding
        query_embedding = embedder.embed_text(request.query)
        
        # Search for relevant documents
        results = opensearch.search(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        # Generate response using LLM
        context = "\n".join([doc["text"] for doc in results])
        answer = llm.generate_response(
            query=request.query,
            context=context
        )
        
        return QueryResponse(
            answer=answer,
            sources=[doc["source"] for doc in results]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 