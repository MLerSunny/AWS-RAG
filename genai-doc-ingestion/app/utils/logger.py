"""
Logging utilities for the application.
"""
import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name (str): Logger name
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Check if logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Try to create file handler if logs directory exists
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Create log file with timestamp
            log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            
            # Add file handler to logger
            logger.addHandler(file_handler)
        except Exception:
            # Continue even if creating file handler fails
            pass
        
        # Add console handler to logger
        logger.addHandler(console_handler)
    
    return logger 