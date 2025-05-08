"""
Unit tests for the SharePoint connector.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Correct import path
from genai_doc_ingestion.app.services.connectors.sharepoint_connector import SharePointConnector


class TestSharePointConnector(unittest.TestCase):
    """Test SharePoint connector."""

    def setUp(self):
        """Set up test case."""
        # Patch imported modules
        self.office365_patch = patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.have_office365', True)
        self.user_cred_patch = patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.UserCredential')
        self.client_ctx_patch = patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.ClientContext')
        
        # Start patches
        self.mock_office365 = self.office365_patch.start()
        self.mock_user_cred = self.user_cred_patch.start()
        self.mock_client_ctx = self.client_ctx_patch.start()
        
        # Set up mock context and web objects
        self.mock_ctx = MagicMock()
        self.mock_web = MagicMock()
        self.mock_ctx.web = self.mock_web
        self.mock_client_ctx.return_value.with_credentials.return_value = self.mock_ctx
        
        # Create connector with test credentials
        self.connector = SharePointConnector(
            site_url="https://test.sharepoint.com/sites/test",
            username="test@example.com",
            password="password123",
            max_workers=2
        )
    
    def tearDown(self):
        """Clean up after test."""
        # Stop patches
        self.office365_patch.stop()
        self.user_cred_patch.stop()
        self.client_ctx_patch.stop()
        
        # Clean up any temporary files
        self.connector.cleanup()
    
    def test_init(self):
        """Test initialization."""
        # Check connector attributes
        self.assertEqual(self.connector.site_url, "https://test.sharepoint.com/sites/test")
        self.assertEqual(self.connector.username, "test@example.com")
        self.assertEqual(self.connector.password, "password123")
        self.assertEqual(self.connector.max_workers, 2)
        self.assertEqual(self.connector.ctx, self.mock_ctx)
        self.assertEqual(self.connector._temp_files, [])
    
    def test_is_connected(self):
        """Test is_connected method."""
        self.assertTrue(self.connector.is_connected())
        
        # Test when ctx is None
        self.connector.ctx = None
        self.assertFalse(self.connector.is_connected())
    
    def test_list_libraries(self):
        """Test list_libraries method."""
        # Mock lists and return value
        mock_list = MagicMock()
        mock_list.title = "Documents"
        mock_lists = [mock_list]
        self.mock_web.lists.filter.return_value.get.return_value.execute_query.return_value = mock_lists
        
        # Call method
        result = self.connector.list_libraries()
        
        # Verify result
        self.assertEqual(result, ["Documents"])
        self.mock_web.lists.filter.assert_called_once_with("BaseTemplate eq 101")
    
    def test_list_libraries_not_connected(self):
        """Test list_libraries when not connected."""
        self.connector.ctx = None
        result = self.connector.list_libraries()
        self.assertEqual(result, [])
    
    @patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.os.path')
    @patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.tempfile')
    def test_download_document(self, mock_tempfile, mock_os_path):
        """Test download_document method."""
        # Set up mocks
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_file.docx"
        mock_tempfile.NamedTemporaryFile.return_value = mock_temp_file
        mock_os_path.splitext.return_value = ("doc", ".docx")
        mock_os_path.basename.return_value = "test_file.docx"
        
        # Mock file object
        mock_file = MagicMock()
        self.mock_web.get_file_by_server_relative_url.return_value = mock_file
        
        # Mock open function
        with patch('builtins.open', mock_open()) as mock_file_open:
            # Call method
            result, filename = self.connector.download_document("Documents/test_file.docx")
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertEqual(filename, "test_file.docx")
            self.assertEqual(self.connector._temp_files, ["/tmp/test_file.docx"])
            self.mock_web.get_file_by_server_relative_url.assert_called_once_with("Documents/test_file.docx")
            mock_file.download.assert_called_once()
    
    def test_download_document_not_connected(self):
        """Test download_document when not connected."""
        self.connector.ctx = None
        result, filename = self.connector.download_document("Documents/test_file.docx")
        self.assertIsNone(result)
        self.assertEqual(filename, "")
    
    def test_batch_download_small_batch(self):
        """Test batch_download with small batch."""
        # Patch download_document method
        with patch.object(self.connector, 'download_document') as mock_download:
            mock_download.side_effect = [
                (MagicMock(), "file1.docx"),
                (MagicMock(), "file2.docx")
            ]
            
            # Call method with small batch
            urls = ["Documents/file1.docx", "Documents/file2.docx"]
            results = self.connector.batch_download(urls)
            
            # Verify results
            self.assertEqual(len(results), 2)
            mock_download.assert_any_call("Documents/file1.docx")
            mock_download.assert_any_call("Documents/file2.docx")
    
    @patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.concurrent.futures.ThreadPoolExecutor')
    def test_batch_download_large_batch(self, mock_executor_class):
        """Test batch_download with large batch (parallel)."""
        # Set up mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Set up mock futures
        mock_future1 = MagicMock()
        mock_future2 = MagicMock()
        mock_future3 = MagicMock()
        
        # Set up return values for futures
        mock_future1.result.return_value = (MagicMock(), "file1.docx")
        mock_future2.result.return_value = (MagicMock(), "file2.docx")
        mock_future3.result.return_value = (MagicMock(), "file3.docx")
        
        # Mock submit method to return futures
        mock_executor.submit.side_effect = [mock_future1, mock_future2, mock_future3]
        
        # Mock as_completed to return futures in order
        with patch('genai_doc_ingestion.app.services.connectors.sharepoint_connector.concurrent.futures.as_completed', 
                  return_value=[mock_future1, mock_future2, mock_future3]):
            
            # Call method with large batch
            urls = ["Documents/file1.docx", "Documents/file2.docx", "Documents/file3.docx"]
            results = self.connector.batch_download(urls)
            
            # Verify results
            self.assertEqual(len(results), 3)
            mock_executor.submit.assert_any_call(self.connector._download_single_document, "Documents/file1.docx")
            mock_executor.submit.assert_any_call(self.connector._download_single_document, "Documents/file2.docx")
            mock_executor.submit.assert_any_call(self.connector._download_single_document, "Documents/file3.docx")
    
    def test_cleanup(self):
        """Test cleanup method."""
        # Add some fake temp files
        self.connector._temp_files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        
        # Mock os.path.exists and os.unlink
        with patch('os.path.exists', return_value=True) as mock_exists:
            with patch('os.unlink') as mock_unlink:
                # Call method
                self.connector.cleanup()
                
                # Verify files were cleaned up
                self.assertEqual(self.connector._temp_files, [])
                self.assertEqual(mock_exists.call_count, 2)
                self.assertEqual(mock_unlink.call_count, 2)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Mock cleanup method
        with patch.object(self.connector, 'cleanup') as mock_cleanup:
            # Use as context manager
            with self.connector as conn:
                # Verify we got back the same object
                self.assertEqual(conn, self.connector)
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main() 