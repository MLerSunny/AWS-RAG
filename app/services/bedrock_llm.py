"""
AWS Bedrock LLM service.
"""
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, Sequence, cast, Callable, Type
import json
import boto3
import time
from botocore.exceptions import ClientError
from ..utils.logger import setup_logger
from ..utils.validation import validate_not_empty, validate_dict, validate_type, validate_range
from ..config import settings
import random

logger = setup_logger(__name__)

class BedrockError(Exception):
    """Base exception for Bedrock client errors."""
    def __init__(self, message: str, error_code: str = "BEDROCK_ERROR", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class ModelError(BedrockError):
    """Raised when model invocation fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", context)

class ValidationError(BedrockError):
    """Raised when input validation fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", context)

class RateLimitError(BedrockError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", context)

class TokenLimitError(BedrockError):
    """Raised when token limit is exceeded."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TOKEN_LIMIT_ERROR", context)

T = TypeVar('T')

class BedrockLLM:
    """Client for interacting with AWS Bedrock LLM models."""
    
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1  # seconds
    MAX_RETRY_DELAY = 10  # seconds
    MAX_TOKENS = 4096  # Maximum tokens supported by most models
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 1.0
    
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
            
        Raises:
            ValidationError: If input parameters are invalid
            BedrockError: If client initialization fails
        """
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.region = region or settings.AWS_REGION
        
        validate_not_empty(self.model_id, "model_id")
        validate_not_empty(self.region, "region")
        validate_range(max_tokens, "max_tokens", 1, self.MAX_TOKENS)
        validate_range(temperature, "temperature", self.MIN_TEMPERATURE, self.MAX_TEMPERATURE)
        
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
            raise BedrockError(f"Failed to initialize Bedrock client: {str(e)}")
    
    def _retry_operation(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Retry an operation with exponential backoff and jitter.
        
        Args:
            operation (Callable[..., Any]): Operation to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Any: Result of the operation
            
        Raises:
            BedrockError: If operation fails after retries
        """
        last_error = None
        delay = self.INITIAL_RETRY_DELAY
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                last_error = e
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                
                # Handle specific error cases
                if error_code == 'ThrottlingException':
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}", {"attempt": attempt + 1})
                elif error_code == 'ValidationException':
                    raise ValidationError(f"Invalid request: {str(e)}", {"attempt": attempt + 1})
                elif error_code == 'ModelStreamErrorException':
                    raise ModelError(f"Model stream error: {str(e)}", {"attempt": attempt + 1})
                
                if attempt == self.MAX_RETRIES - 1:
                    break
                    
                # Add jitter to delay
                jitter = (0.5 + random.random()) * delay
                delay = min(delay * 2, self.MAX_RETRY_DELAY)
                
                logger.warning(f"Operation failed, retrying in {jitter:.2f}s: {str(e)}")
                time.sleep(jitter)
            except Exception as e:
                raise BedrockError(f"Operation failed: {str(e)}")
        
        raise BedrockError(f"Operation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    def is_initialized(self) -> bool:
        """
        Check if the client is initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.client is not None
    
    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build the prompt based on model type.
        
        Args:
            query (str): The query to answer
            context (Optional[str]): Context to include for RAG
            system_prompt (Optional[str]): System prompt for the model
            
        Returns:
            Dict[str, Any]: Model-specific prompt structure
        """
        if "anthropic.claude" in self.model_id:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            user_content = query
            if context:
                user_content = f"Context: {context}\n\nQuestion: {query}"
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages
            }
        elif "amazon.titan" in self.model_id:
            prompt = query
            if context:
                prompt = f"Context: {context}\n\nQuestion: {query}"
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": self.temperature
                }
            }
        elif "meta.llama" in self.model_id:
            prompt = query
            if context:
                prompt = f"Context: {context}\n\nQuestion: {query}"
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            
            return {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        else:
            raise ModelError(f"Unsupported model: {self.model_id}")
    
    def _parse_response(self, response: Dict[str, Any]) -> str:
        """
        Parse the model response based on model type.
        
        Args:
            response (Dict[str, Any]): Raw model response
            
        Returns:
            str: Extracted text response
        """
        if "anthropic.claude" in self.model_id:
            return response["content"][0]["text"]
        elif "amazon.titan" in self.model_id:
            return response["results"][0]["outputText"]
        elif "meta.llama" in self.model_id:
            return response["generation"]
        else:
            raise ModelError(f"Unsupported model: {self.model_id}")
    
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
            
        Raises:
            ValidationError: If input parameters are invalid
            ModelError: If model invocation fails
        """
        validate_not_empty(query, "query")
        if max_tokens is not None:
            validate_range(max_tokens, "max_tokens", 1, self.MAX_TOKENS)
        if temperature is not None:
            validate_range(temperature, "temperature", self.MIN_TEMPERATURE, self.MAX_TEMPERATURE)
        if stop_sequences is not None:
            validate_type(stop_sequences, list, "stop_sequences")
            for seq in stop_sequences:
                validate_not_empty(seq, "stop_sequence")
        
        if not self.is_initialized():
            raise BedrockError("Bedrock client not initialized")
            
        try:
            # Use provided values or defaults
            self.max_tokens = max_tokens or self.max_tokens
            self.temperature = temperature or self.temperature
            
            # Build prompt based on model type
            prompt = self._build_prompt(query, context, system_prompt)
            
            # Add stop sequences if provided
            if stop_sequences:
                prompt["stop_sequences"] = stop_sequences
            
            # Invoke model
            response = self._retry_operation(
                self.client.invoke_model,
                modelId=self.model_id,
                body=json.dumps(prompt)
            )
            
            # Parse response
            return self._parse_response(json.loads(response["body"].read()))
            
        except Exception as e:
            if isinstance(e, (ValidationError, ModelError, RateLimitError, TokenLimitError)):
                raise
            logger.error(f"Error generating response: {str(e)}")
            raise ModelError(f"Failed to generate response: {str(e)}") 