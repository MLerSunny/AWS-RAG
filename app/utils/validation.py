"""
Validation utilities for input parameters.
"""
from typing import Any, Dict, Optional
from app.utils.logger import setup_logger
import re
import os

logger = setup_logger(__name__)

def validate_not_empty(value: Any, param_name: str) -> None:
    """
    Validate that a value is not empty.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If the value is empty
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{param_name} cannot be empty")
    
    if isinstance(value, (list, dict, set)) and not value:
        raise ValueError(f"{param_name} cannot be empty")

def validate_dict(value: Any, param_name: str) -> None:
    """
    Validate that a value is a dictionary.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If the value is not a dictionary
    """
    if not isinstance(value, dict):
        raise ValueError(f"{param_name} must be a dictionary")

def validate_type(value: Any, expected_type: type, param_name: str) -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: The value to validate
        expected_type: The expected type
        param_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        raise ValueError(f"{param_name} must be of type {expected_type.__name__}")

def validate_range(value: float, param_name: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> None:
    """
    Validate that a numeric value is within a range.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Raises:
        ValueError: If the value is outside the allowed range
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be greater than or equal to {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be less than or equal to {max_value}")

def validate_query(query: str, min_length: int = 3) -> bool:
    """
    Validate query text.
    
    Args:
        query: Query text to validate
        min_length: Minimum allowed length
        
    Returns:
        True if query is valid, False otherwise
    """
    if not query or len(query.strip()) < min_length:
        return False
        
    # Remove common punctuation and whitespace
    cleaned = re.sub(r'[^\w\s]', '', query).strip()
    
    # Check if query is too short after cleaning
    if len(cleaned) < min_length:
        return False
        
    # Check for common spam patterns
    spam_patterns = [
        r'^\d+$',  # Just numbers
        r'^[a-zA-Z]{1,2}$',  # Very short words
        r'^[^a-zA-Z0-9]+$',  # No alphanumeric
        r'(.)\1{4,}',  # Repeated characters
    ]
    
    for pattern in spam_patterns:
        if re.match(pattern, cleaned):
            return False
            
    return True

def sanitize_query(query: str) -> Optional[str]:
    """
    Sanitize query text.
    
    Args:
        query: Query text to sanitize
        
    Returns:
        Sanitized query or None if invalid
    """
    if not validate_query(query):
        return None
        
    # Remove special characters but keep spaces
    cleaned = re.sub(r'[^\w\s]', '', query)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def validate_filename(filename: str) -> bool:
    """
    Validate filename.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if filename is valid, False otherwise
    """
    if not filename:
        return False
        
    # Check length
    if len(filename) > 255:  # Maximum filename length
        return False
        
    # Check for invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    if re.search(invalid_chars, filename):
        return False
        
    # Check for reserved names
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4',
        'LPT1', 'LPT2', 'LPT3', 'LPT4'
    }
    name = os.path.splitext(filename)[0].upper()
    if name in reserved_names:
        return False
        
    return True 