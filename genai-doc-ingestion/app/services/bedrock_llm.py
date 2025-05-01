"""
AWS Bedrock LLM service.
"""
from typing import Dict, List, Optional, Any, Union
import json
import boto3
import time
from ..utils.logger import setup_logger
from ..config import settings

logger = setup_logger(__name__)

class BedrockLLM:
    """Client for interacting with AWS Bedrock LLM models."""
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize the Bedrock LLM client.
        
        Args:
            model_id (Optional[str]): Bedrock model ID
            region (Optional[str]): AWS region
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for text generation
        """
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.region = region or settings.AWS_REGION
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        try:
            # Initialize AWS Bedrock client
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"Initialized Bedrock client for model {self.model_id}")
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {str(e)}")
            self.client = None
    
    def is_initialized(self) -> bool:
        """
        Check if the client is initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.client is not None
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            query (str): The query to answer
            context (Optional[str]): Context to include for RAG
            system_prompt (Optional[str]): System prompt for the model
            max_tokens (Optional[int]): Maximum tokens to generate
            temperature (Optional[float]): Temperature for generation
            stop_sequences (Optional[List[str]]): Sequences to stop generation
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized():
            logger.error("Bedrock client not initialized")
            return "Error: Bedrock client not initialized"
            
        try:
            # Use provided values or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Build prompt based on model type
            if "anthropic.claude" in self.model_id:
                response = self._invoke_claude(
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences
                )
            elif "amazon.titan" in self.model_id:
                response = self._invoke_titan(
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences
                )
            elif "meta.llama" in self.model_id:
                response = self._invoke_llama(
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences
                )
            else:
                logger.error(f"Unsupported model: {self.model_id}")
                return f"Error: Unsupported model {self.model_id}"
                
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _invoke_claude(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Invoke Anthropic Claude model.
        
        Args:
            query (str): The query to answer
            context (Optional[str]): Context to include for RAG
            system_prompt (Optional[str]): System prompt for the model
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for generation
            stop_sequences (Optional[List[str]]): Sequences to stop generation
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized():
            return "Error: Bedrock client not initialized"
            
        # Build messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message with context if provided
        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {query}"
            
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Build request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences
        # Invoke model
        try:
            if not self.client:
                raise AttributeError("Bedrock client is None")
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
        except AttributeError:
            return "Error: Failed to invoke Bedrock model - client not properly initialized"
        # Parse response
        response_body = json.loads(response.get("body").read())
        return response_body.get("content", [{}])[0].get("text", "")
    
    def _invoke_titan(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Invoke Amazon Titan model.
        
        Args:
            query (str): The query to answer
            context (Optional[str]): Context to include for RAG
            system_prompt (Optional[str]): System prompt for the model
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for generation
            stop_sequences (Optional[List[str]]): Sequences to stop generation
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized():
            return "Error: Bedrock client not initialized"
            
        # Build prompt
        prompt = query
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Build request body
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            request_body["textGenerationConfig"]["stopSequences"] = stop_sequences
        # Invoke model
        try:
            if not self.client:
                raise AttributeError("Bedrock client is None")
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
        except AttributeError:
            return "Error: Bedrock client not initialized properly"
        # Parse response
        response_body = json.loads(response.get("body").read())
        return response_body.get("results", [{}])[0].get("outputText", "")
    
    def _invoke_llama(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Invoke Meta Llama model.
        
        Args:
            query (str): The query to answer
            context (Optional[str]): Context to include for RAG
            system_prompt (Optional[str]): System prompt for the model
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for generation
            stop_sequences (Optional[List[str]]): Sequences to stop generation
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized():
            return "Error: Bedrock client not initialized"
            
        # Build prompt
        prompt = query
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Build request body
        request_body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_body["system_prompt"] = system_prompt
            
        # Stop sequences not directly supported in Llama models
        # Invoke model
        try:
            if not self.client:
                raise AttributeError("Bedrock client is None")
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
        except AttributeError:
            return "Error: Bedrock client not initialized properly"
        # Parse response
        response_body = json.loads(response.get("body").read())
        return response_body.get("generation", "") 