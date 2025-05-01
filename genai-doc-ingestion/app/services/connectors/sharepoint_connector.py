"""
SharePoint connector for retrieving documents from SharePoint sites.
"""
import os
import tempfile
import concurrent.futures
from typing import List, Dict, Optional, Any, Tuple, BinaryIO, Union, cast
from ...utils.logger import setup_logger
from ...config import settings

# Conditional imports to handle missing packages
try:
    from office365.runtime.auth.user_credential import UserCredential
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File
    have_office365 = True
except ImportError:
    have_office365 = False
    UserCredential = None
    ClientContext = None
    File = None

logger = setup_logger(__name__)

class SharePointConnector:
    """Connector to retrieve documents from SharePoint sites."""
    
    def __init__(
        self, 
        site_url: Optional[str] = None, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        max_workers: int = 4
    ):
        """
        Initialize the SharePoint connector.
        
        Args:
            site_url (str): The SharePoint site URL
            username (str): SharePoint username
            password (str): SharePoint password
            max_workers (int): Maximum number of parallel workers for batch operations
        """
        self.site_url = site_url or settings.SHAREPOINT_SITE_URL
        self.username = username or settings.SHAREPOINT_USERNAME
        self.password = password or settings.SHAREPOINT_PASSWORD
        self.max_workers = max_workers
        self.ctx = None
        self._temp_files = []
        
        if not all([self.site_url, self.username, self.password]):
            logger.warning("SharePoint credentials not fully configured")
        else:
            try:
                # Check if Office365 packages are available
                if not have_office365:
                    logger.error("Office365 packages are not installed. Please install with 'pip install Office365-REST-Python-Client'")
                    return
                
                # Initialize SharePoint client
                if (UserCredential is not None and 
                    ClientContext is not None and 
                    isinstance(self.site_url, str) and 
                    isinstance(self.username, str) and 
                    isinstance(self.password, str)):
                    
                    user_credentials = UserCredential(self.username, self.password)
                    self.ctx = ClientContext(self.site_url).with_credentials(user_credentials)
                    logger.info(f"SharePoint client initialized for {self.site_url}")
            except Exception as e:
                logger.error(f"Error initializing SharePoint client: {str(e)}")
                self.ctx = None
    
    def __enter__(self):
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method that cleans up resources.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def cleanup(self):
        """Clean up any temporary files created during operations."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")
        
        self._temp_files = []
    
    def is_connected(self) -> bool:
        """
        Check if the connector is properly initialized.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.ctx is not None
    
    def list_libraries(self) -> List[str]:
        """
        List available document libraries.
        
        Returns:
            List[str]: List of document library names
        """
        if not self.is_connected():
            logger.error("SharePoint client not initialized")
            return []
            
        try:
            # Get document libraries
            if self.ctx and have_office365 and self.ctx.web:
                lists = self.ctx.web.lists.filter("BaseTemplate eq 101").get().execute_query()
                return [lib.title for lib in lists if lib.title]
            return []
        except Exception as e:
            logger.error(f"Error listing SharePoint libraries: {str(e)}")
            return []
    
    def list_documents(self, library_name: str, folder_path: str = "") -> List[Dict[str, Any]]:
        """
        List documents in a library/folder.
        
        Args:
            library_name (str): Name of the document library
            folder_path (str): Optional subfolder path
            
        Returns:
            List[Dict]: List of document metadata
        """
        if not self.is_connected():
            logger.error("SharePoint client not initialized")
            return []
            
        try:
            # Build folder path
            if folder_path:
                folder_url = f"{library_name}/{folder_path}"
            else:
                folder_url = library_name
                
            # Get folder and files
            if self.ctx and have_office365 and self.ctx.web:
                folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
                files = folder.files.get().execute_query()
                
                # Format results
                results = []
                for file in files:
                    created_time = None
                    if file.time_created:
                        created_time = file.time_created.strftime("%Y-%m-%d %H:%M:%S")
                        
                    modified_time = None
                    if file.time_last_modified:
                        modified_time = file.time_last_modified.strftime("%Y-%m-%d %H:%M:%S")
                    
                    results.append({
                        "name": file.name,
                        "url": file.serverRelativeUrl,
                        "size": file.length,
                        "created": created_time,
                        "modified": modified_time
                    })
                
                return results
            return []
        except Exception as e:
            logger.error(f"Error listing SharePoint documents: {str(e)}")
            return []
    
    def download_document(self, document_url: str) -> Tuple[Optional[BinaryIO], str]:
        """
        Download a document from SharePoint.
        
        Args:
            document_url (str): The document URL
            
        Returns:
            Tuple[Optional[BinaryIO], str]: Tuple of (file object, filename)
        """
        if not self.is_connected():
            logger.error("SharePoint client not initialized")
            return None, ""
            
        try:
            # Get file by server relative URL
            if self.ctx and self.ctx.web:
                file_obj = self.ctx.web.get_file_by_server_relative_url(document_url)
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(document_url)[-1])
                temp_file_path = temp_file.name
                self._temp_files.append(temp_file_path)  # Track for cleanup
                
                # Download content
                with open(temp_file_path, "wb") as local_file:
                    file_obj.download(local_file).execute_query()
                
                # Return file object and filename
                filename = os.path.basename(document_url)
                
                # Return the file opened in binary read mode and the filename
                return open(temp_file_path, "rb"), filename
            return None, ""
        except Exception as e:
            logger.error(f"Error downloading SharePoint document: {str(e)}")
            return None, ""
    
    def _download_single_document(self, document_url: str) -> Tuple[Optional[BinaryIO], str]:
        """
        Internal method to download a single document for parallel processing.
        
        Args:
            document_url (str): The document URL
            
        Returns:
            Tuple[Optional[BinaryIO], str]: Tuple of (file object, filename)
        """
        try:
            return self.download_document(document_url)
        except Exception as e:
            logger.error(f"Error in parallel download of {document_url}: {str(e)}")
            return None, ""
    
    def batch_download(self, document_urls: List[str]) -> List[Tuple[Optional[BinaryIO], str]]:
        """
        Download multiple documents from SharePoint in parallel.
        
        Args:
            document_urls (List[str]): List of document URLs
            
        Returns:
            List[Tuple[Optional[BinaryIO], str]]: List of (file object, filename) tuples
        """
        if not document_urls:
            return []
        
        # For small batches, just use sequential download
        if len(document_urls) <= 2:
            return [self.download_document(url) for url in document_urls]
        
        # For larger batches, use parallel processing
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self._download_single_document, url): url 
                for url in document_urls
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Exception downloading {url}: {str(e)}")
                    results.append((None, ""))
        
        return results 