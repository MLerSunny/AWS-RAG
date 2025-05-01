"""
SharePoint connector for retrieving documents from SharePoint sites.
"""
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
import os
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class SharePointConnector:
    def __init__(self, site_url, username, password):
        """
        Initialize SharePoint connector with authentication details.
        
        Args:
            site_url (str): The SharePoint site URL
            username (str): SharePoint username
            password (str): SharePoint password
        """
        try:
            auth_context = AuthenticationContext(site_url)
            auth_context.acquire_token_for_user(username, password)
            self.ctx = ClientContext(site_url, auth_context)
            logger.info(f"Successfully initialized SharePoint connector for {site_url}")
        except Exception as e:
            logger.error(f"Error initializing SharePoint connector: {str(e)}")
            raise
    
    def download_document(self, document_url, local_path):
        """
        Download a document from SharePoint to a local path.
        
        Args:
            document_url (str): The server-relative URL of the document
            local_path (str): The local path to save the document
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, "wb") as local_file:
                file = self.ctx.web.get_file_by_server_relative_url(document_url)
                file.download(local_file).execute_query()
            
            logger.info(f"Successfully downloaded {document_url} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading document {document_url}: {str(e)}")
            raise
    
    def list_documents(self, folder_url):
        """
        List all documents in a SharePoint folder.
        
        Args:
            folder_url (str): The server-relative URL of the folder
            
        Returns:
            list: List of document details
        """
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()
            
            results = []
            for file in files:
                results.append({
                    "name": file.properties["Name"],
                    "url": file.properties["ServerRelativeUrl"],
                    "size": file.properties["Length"],
                    "modified": file.properties["TimeLastModified"]
                })
            
            logger.info(f"Found {len(results)} documents in {folder_url}")
            return results
        except Exception as e:
            logger.error(f"Error listing documents in {folder_url}: {str(e)}")
            raise 