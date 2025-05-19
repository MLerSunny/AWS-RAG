"""
Common utility functions shared across multiple modules.
"""
import time
import math
import logging
from typing import Any, Dict, Callable, Optional
from functools import wraps

# Setup logger
logger = logging.getLogger(__name__)

def execute_with_retry(operation_func: Callable, *args, max_retries: int = 3, 
                      retry_delay: float = 1.0, **kwargs) -> Any:
    """
    Execute an operation with retry logic.
    
    Args:
        operation_func: Function to execute
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (seconds)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: Result of the operation function
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            retries += 1
            
            if retries <= max_retries:
                # Exponential backoff
                sleep_time = retry_delay * (2 ** (retries - 1))
                logger.warning(f"Operation failed, retrying in {sleep_time:.2f}s ({retries}/{max_retries}): {str(e)}")
                time.sleep(sleep_time)
    
    # If we've exhausted retries, log and re-raise the last exception
    logger.error(f"Operation failed after {max_retries} retries: {str(last_exception)}")
    return None

def retry_decorator(max_retries: int = 3, retry_delay: float = 1.0):
    """
    Decorator for applying retry logic to any function.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (seconds)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return execute_with_retry(
                func, *args, max_retries=max_retries, 
                retry_delay=retry_delay, **kwargs
            )
        return wrapper
    return decorator

def compute_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float], 
                             use_numpy: bool = True) -> float:
    """
    Compute cosine similarity between two term frequency vectors.
    
    Args:
        vec1: First term frequency vector
        vec2: Second term frequency vector
        use_numpy: Whether to use numpy for vectorized calculation
        
    Returns:
        float: Cosine similarity score
    """
    # Use numpy for vectorized calculation if available and requested
    if use_numpy:
        try:
            import numpy as np
            
            # Get all unique terms
            all_terms = set(vec1.keys()) | set(vec2.keys())
            
            # Create vectors
            v1 = np.array([vec1.get(term, 0.0) for term in all_terms])
            v2 = np.array([vec2.get(term, 0.0) for term in all_terms])
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(np.dot(v1, v2) / (norm1 * norm2))
        except (ImportError, Exception) as e:
            logger.warning(f"Error in numpy cosine calculation: {str(e)}")
            # Fall back to non-vectorized calculation
    
    # Manual calculation as fallback
    try:
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
            
        # Calculate numerator (dot product)
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate denominators (magnitudes)
        magnitude1 = math.sqrt(sum(vec1[term] ** 2 for term in vec1))
        magnitude2 = math.sqrt(sum(vec2[term] ** 2 for term in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        logger.warning(f"Error in cosine similarity calculation: {str(e)}")
        return 0.0 