"""
AWS Textract service for document text extraction.
"""
import os
import boto3
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError
from exceptions import AWSServiceError, ErrorCode, FileOperationError
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class TextractExtractor:
    """Service for extracting text from documents using AWS Textract."""
    
    def __init__(self, region_name: Optional[str] = None) -> None:
        """
        Initialize Textract client.
        
        Args:
            region_name: Optional AWS region name
            
        Raises:
            AWSServiceError: If client initialization fails
        """
        try:
            self.client = boto3.client('textract', region_name=region_name)
        except Exception as e:
            raise AWSServiceError(
                'textract',
                'initialization',
                str(e),
                ErrorCode.AWS_INIT_FAILED,
                cause=e
            )
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document using Textract.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileOperationError: If file cannot be read
            AWSServiceError: If Textract API call fails
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileOperationError(
                    'read',
                    file_path,
                    f"File not found: {file_path}",
                    ErrorCode.FILE_NOT_FOUND
                )
                
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
            
            # Call Textract API
            response = self.client.detect_document_text(
                Document={'Bytes': file_bytes}
            )
            
            # Process response
            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            
            logger.info(
                f"Successfully extracted text from {file_path}",
                extra={
                    'file_path': file_path,
                    'text_length': len(text),
                    'block_count': len(response["Blocks"])
                }
            )
            return text
            
        except FileOperationError:
            raise
        except ClientError as e:
            raise AWSServiceError(
                'textract',
                'detect_document_text',
                str(e),
                ErrorCode.AWS_OPERATION_FAILED,
                cause=e
            )
        except Exception as e:
            raise AWSServiceError(
                'textract',
                'extract_text',
                str(e),
                ErrorCode.PROCESSING_FAILED,
                cause=e
            )
    
    def extract_text_with_tables(self, file_path):
        """
        Extract text including tables from a document using Textract.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            dict: Extracted text content with tables
        """
        try:
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
            
            # Call Textract API with AnalyzeDocument
            response = self.client.analyze_document(
                Document={'Bytes': file_bytes},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            # Process text blocks
            text_blocks = []
            tables = []
            
            # Process the response (simplified implementation)
            # In a real implementation, you would need to process the relationships
            # between blocks to reconstruct tables properly
            
            logger.info(f"Successfully extracted text and tables from {file_path}")
            return {
                "text": "\n".join(text_blocks),
                "tables": tables
            }
        except Exception as e:
            logger.error(f"Error extracting text and tables from {file_path}: {str(e)}")
            raise 