"""
Extract text from documents using AWS Textract service.
"""
import boto3
import os
from ...config import settings
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class TextractExtractor:
    def __init__(self, region_name=None):
        """
        Initialize the Textract extractor.
        
        Args:
            region_name (str, optional): AWS region name, defaults to settings.AWS_REGION
        """
        region = region_name or settings.AWS_REGION
        self.client = boto3.client(
            'textract', 
            region_name=region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        logger.info(f"Initialized Textract extractor in region {region}")
    
    def extract_text(self, file_path):
        """
        Extract text from a document using Textract.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
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
            
            logger.info(f"Successfully extracted text from {file_path} ({len(text)} characters)")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
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