"""
SharePoint connector for retrieving documents from SharePoint sites.
"""
import os
import tempfile
import concurrent.futures
from typing import List, Dict, Optional, Any, Tuple, BinaryIO, TYPE_CHECKING, cast, Type, Union
from datetime import datetime
from exceptions import (
    ConfigurationError, ProcessingError, ErrorCode,
    FileOperationError
)
from ...utils.logger import setup_logger
from ...utils.common import execute_with_retry

if TYPE_CHECKING:
    from office365.runtime.auth.client_credential import ClientCredential
    from office365.sharepoint.client_context import ClientContext

# Conditional import to handle missing package
try:
    from office365.runtime.auth.client_credential import ClientCredential
    from office365.sharepoint.client_context import ClientContext
    HAVE_OFFICE365 = True
except ImportError:
    ClientCredential = None
    ClientContext = None
    HAVE_OFFICE365 = False

logger = setup_logger(__name__)

class SharePointConnector:
    """Connector to retrieve documents from SharePoint sites."""
    
    def __init__(
        self, 
        site_url: Optional[str] = None, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        max_workers: int = 4
    ) -> None:
        """
        Initialize SharePoint connector.
        
        Args:
            site_url: SharePoint site URL
            username: SharePoint username
            password: SharePoint password
            max_workers: Maximum number of concurrent workers
            
        Raises:
            ConfigurationError: If required configuration is missing
            ImportError: If office365 package is not installed
        """
        if not HAVE_OFFICE365:
            raise ImportError("office365 package is required but not installed")
            
        self.site_url = site_url or os.environ.get("SHAREPOINT_SITE_URL")
        self.username = username or os.environ.get("SHAREPOINT_USERNAME")
        self.password = password or os.environ.get("SHAREPOINT_PASSWORD")
        
        if not all([self.site_url, self.username, self.password]):
            raise ConfigurationError(
                "Missing required SharePoint configuration",
                context={
                    'has_url': bool(self.site_url),
                    'has_username': bool(self.username),
                    'has_password': bool(self.password)
                }
            )
            
        self.max_workers = max_workers
        self.ctx: Union[ClientContext, None] = None
        self._temp_files: List[str] = []
        self._last_connection: Optional[datetime] = None
        self._connection_timeout = 3600  # 1 hour
        
    def connect(self) -> None:
        """
        Establish connection to SharePoint.
        
        Raises:
            ProcessingError: If connection fails
        """
        if self.is_connected() and not self._is_connection_expired():
            return
            
        if not self.site_url or not self.username or not self.password:
            raise ProcessingError("Missing required SharePoint credentials")
            
        try:
            if not HAVE_OFFICE365 or not ClientCredential or not ClientContext:
                raise ImportError("Office365 packages are not installed")
                
            credentials = ClientCredential(self.username, self.password)
            self.ctx = ClientContext(self.site_url).with_credentials(credentials)
            self._last_connection = datetime.now()
            logger.info(f"Connected to SharePoint site: {self.site_url}")
        except Exception as e:
            raise ProcessingError(
                f"Failed to connect to SharePoint: {str(e)}",
                context={'site_url': self.site_url},
                cause=e
            )
            
    def _is_connection_expired(self) -> bool:
        """Check if the current connection has expired."""
        if not self._last_connection:
            return True
        return (datetime.now() - self._last_connection).total_seconds() > self._connection_timeout
        
    def is_connected(self) -> bool:
        """
        Check if the connector is properly connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.ctx is not None
        
    def disconnect(self) -> None:
        """Disconnect from SharePoint and clean up resources."""
        self.ctx = None
        self._last_connection = None
        self.cleanup()
        logger.info("Disconnected from SharePoint")
        
    def __enter__(self) -> 'SharePointConnector':
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
        
    def cleanup(self) -> None:
        """Clean up any temporary files created during operations."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")
        
        self._temp_files = []
        
    def list_libraries(self) -> List[str]:
        """
        List available document libraries.
        
        Returns:
            List[str]: List of document library names
            
        Raises:
            ProcessingError: If listing libraries fails
        """
        if not self.is_connected():
            self.connect()
            
        if not self.ctx:
            raise ProcessingError("SharePoint client not initialized")
            
        try:
            lists = self.ctx.web.lists.filter("BaseTemplate eq 101").get().execute_query()
            return [lib.title for lib in lists if lib.title]
        except Exception as e:
            raise ProcessingError(
                f"Failed to list SharePoint libraries: {str(e)}",
                context={'site_url': self.site_url},
                cause=e
            )
            
    def list_documents(self, library_name: str, folder_path: str = "") -> List[Dict[str, Any]]:
        """
        List documents in a library/folder.
        
        Args:
            library_name: Name of the document library
            folder_path: Optional subfolder path
            
        Returns:
            List[Dict]: List of document metadata
            
        Raises:
            ProcessingError: If listing documents fails
        """
        if not self.is_connected():
            self.connect()
            
        if not self.ctx:
            raise ProcessingError("SharePoint client not initialized")
            
        try:
            library = self.ctx.web.lists.get_by_title(library_name)
            items = library.items.get().execute_query()
            
            documents = []
            for item in items:
                if item.properties.get('FSObjType') == 0:  # File
                    documents.append({
                        'name': item.properties.get('FileLeafRef'),
                        'url': item.properties.get('FileRef'),
                        'size': item.properties.get('File_x0020_Size'),
                        'created': item.properties.get('Created'),
                        'modified': item.properties.get('Modified')
                    })
            return documents
        except Exception as e:
            raise ProcessingError(
                f"Failed to list documents in {library_name}: {str(e)}",
                context={
                    'library_name': library_name,
                    'folder_path': folder_path
                },
                cause=e
            )
            
    def download_document(self, document_url: str) -> Tuple[Optional[BinaryIO], str]:
        """
        Download a document from SharePoint.
        
        Args:
            document_url: The document URL
            
        Returns:
            Tuple[Optional[BinaryIO], str]: Tuple of (file object, filename)
            
        Raises:
            ProcessingError: If download fails
        """
        if not self.is_connected():
            self.connect()
            
        if not self.ctx:
            raise ProcessingError("SharePoint client not initialized")
            
        try:
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
            return open(temp_file_path, "rb"), filename
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to download document {document_url}: {str(e)}",
                context={'document_url': document_url},
                cause=e
            )
            
    def _download_single_document(self, document_url: str) -> Tuple[Optional[BinaryIO], str]:
        """
        Download a single document with retry logic.
        
        Args:
            document_url: The document URL
            
        Returns:
            Tuple[Optional[BinaryIO], str]: Tuple of (file object, filename)
        """
        return execute_with_retry(
            lambda: self.download_document(document_url),
            max_retries=3,
            error_codes={ProcessingError: 1}  # Retry on any ProcessingError
        )
        
    def download_documents(self, document_urls: List[str]) -> List[Tuple[Optional[BinaryIO], str]]:
        """
        Download multiple documents in parallel.
        
        Args:
            document_urls: List of document URLs
            
        Returns:
            List[Tuple[Optional[BinaryIO], str]]: List of (file object, filename) tuples
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._download_single_document, url)
                for url in document_urls
            ]
            return [future.result() for future in concurrent.futures.as_completed(futures)] 