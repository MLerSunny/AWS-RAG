import json
import boto3
from typing import Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class BedrockLLM:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = settings.BEDROCK_MODEL_ID
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate a response using Claude with RAG context."""
        try:
            prompt = f"""Human: You are a helpful AI assistant. Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Assistant:"""
            
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 1000,
                "temperature": 0.5,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:"]
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            response_body = json.loads(response.get("body").read())
            return response_body.get("completion", "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 