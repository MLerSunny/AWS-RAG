"""
ServiceNow connector for retrieving tickets and knowledge base articles.
"""
import os
import json
import tempfile
import time
from typing import List, Dict, Optional, Any, Union, TypeVar, Generic, cast, TYPE_CHECKING
from functools import lru_cache
from datetime import datetime
from exceptions import (
    ConfigurationError, ProcessingError, ErrorCode,
    FileOperationError
)
from ...utils.logger import setup_logger
from ...utils.common import execute_with_retry

if TYPE_CHECKING:
    from pysnow import Client

# Conditional import to handle missing package
try:
    import pysnow as snow
    from pysnow import Client as SnowClient
    HAVE_SNOW = True
except ImportError:
    snow = None
    SnowClient = None
    HAVE_SNOW = False

logger = setup_logger(__name__)

T = TypeVar('T')

class ServiceNowConnector:
    """Connector to retrieve data from ServiceNow instances."""
    
    def __init__(
        self, 
        instance_url: Optional[str] = None, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        """
        Initialize the ServiceNow connector.
        
        Args:
            instance_url: ServiceNow instance URL
            username: ServiceNow username
            password: ServiceNow password
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            
        Raises:
            ConfigurationError: If required configuration is missing
            ImportError: If pysnow package is not installed
        """
        if not HAVE_SNOW:
            raise ImportError("pysnow package is required but not installed")
            
        self.instance_url = instance_url or os.environ.get("SERVICENOW_INSTANCE_URL")
        self.username = username or os.environ.get("SERVICENOW_USERNAME")
        self.password = password or os.environ.get("SERVICENOW_PASSWORD")
        
        if not all([self.instance_url, self.username, self.password]):
            raise ConfigurationError(
                "Missing required ServiceNow configuration",
                context={
                    'has_url': bool(self.instance_url),
                    'has_username': bool(self.username),
                    'has_password': bool(self.password)
                },
                error_code=ErrorCode.CONFIG_MISSING
            )
            
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client: Optional[SnowClient] = None
        self._cache = {}
        self._last_connection = None
        self._connection_timeout = 3600  # 1 hour
        
    def connect(self) -> None:
        """
        Establish connection to ServiceNow.
        
        Raises:
            ProcessingError: If connection fails
        """
        if self.is_connected() and not self._is_connection_expired():
            return
            
        try:
            self.client = snow.Client(
                instance=self.instance_url,
                user=self.username,
                password=self.password
            )
            self._last_connection = datetime.now()
            logger.info(f"Connected to ServiceNow instance: {self.instance_url}")
        except Exception as e:
            raise ProcessingError(
                f"Failed to connect to ServiceNow: {str(e)}",
                context={'instance_url': self.instance_url},
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
        return self.client is not None
        
    def disconnect(self) -> None:
        """Disconnect from ServiceNow."""
        self.client = None
        self._last_connection = None
        self._cache.clear()
        logger.info("Disconnected from ServiceNow")
        
    def __enter__(self) -> 'ServiceNowConnector':
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
        
    @lru_cache(maxsize=100)
    def get_ticket(self, ticket_number: str) -> Dict[str, Any]:
        """
        Get ticket details by number.
        
        Args:
            ticket_number: ServiceNow ticket number
            
        Returns:
            Dict containing ticket details
            
        Raises:
            ProcessingError: If ticket retrieval fails
        """
        if not self.is_connected():
            self.connect()
            
        if not self.client:
            raise ProcessingError("ServiceNow client not initialized")
            
        try:
            resource = self.client.resource('api/now/table/incident')
            response = resource.get(query={'number': ticket_number})
            return response.one()
        except Exception as e:
            raise ProcessingError(
                f"Failed to retrieve ticket {ticket_number}: {str(e)}",
                context={'ticket_number': ticket_number},
                cause=e
            )
            
    def get_kb_article(self, article_number: str) -> Dict[str, Any]:
        """
        Get knowledge base article by number.
        
        Args:
            article_number: KB article number
            
        Returns:
            Dict containing article details
            
        Raises:
            ProcessingError: If article retrieval fails
        """
        if not self.is_connected():
            self.connect()
            
        if not self.client:
            raise ProcessingError("ServiceNow client not initialized")
            
        try:
            resource = self.client.resource('api/now/table/kb_knowledge')
            response = resource.get(query={'number': article_number})
            return response.one()
        except Exception as e:
            raise ProcessingError(
                f"Failed to retrieve KB article {article_number}: {str(e)}",
                context={'article_number': article_number},
                cause=e
            )
    
    def _execute_with_retry(self, operation_func, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the operation function
        """
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries <= self.max_retries:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(f"API call failed, retrying in {sleep_time:.2f}s ({retries}/{self.max_retries}): {str(e)}")
                    time.sleep(sleep_time)
        
        # If we've exhausted retries, log and re-raise the last exception
        logger.error(f"API call failed after {self.max_retries} retries: {str(last_exception)}")
        return []
    
    @lru_cache(maxsize=100)
    def get_incidents(self, query_str: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve incidents from ServiceNow with caching.
        
        Args:
            query_str (str): Query string (converted to dict internally)
            limit (int): Maximum number of incidents to retrieve
            
        Returns:
            List[Dict]: List of incidents
        """
        if not self.is_connected():
            logger.error("ServiceNow client not initialized")
            return []
        
        # Convert query string to dict for caching purposes
        query = {"active": "true"}
        if query_str:
            try:
                query = json.loads(query_str)
            except json.JSONDecodeError:
                logger.error(f"Invalid query string: {query_str}")
                
        def _fetch_incidents():
            if self.client:
                incident = self.client.resource(api_path='/table/incident')
                response = incident.get(query=query, limit=limit)
                results = list(response.all())
                logger.info(f"Retrieved {len(results)} incidents from ServiceNow")
                return results
            return []
                
        return self._execute_with_retry(_fetch_incidents)
    
    @lru_cache(maxsize=100)
    def get_knowledge_articles(self, query_str: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge base articles from ServiceNow with caching.
        
        Args:
            query_str (str): Query string (converted to dict internally)
            limit (int): Maximum number of articles to retrieve
            
        Returns:
            List[Dict]: List of knowledge articles
        """
        if not self.is_connected():
            logger.error("ServiceNow client not initialized")
            return []
        
        # Convert query string to dict for caching purposes
        query = {"active": "true", "workflow_state": "published"}
        if query_str:
            try:
                query = json.loads(query_str)
            except json.JSONDecodeError:
                logger.error(f"Invalid query string: {query_str}")
                
        def _fetch_articles():
            if self.client:
                kb = self.client.resource(api_path='/table/kb_knowledge')
                response = kb.get(query=query, limit=limit)
                results = list(response.all())
                logger.info(f"Retrieved {len(results)} knowledge articles from ServiceNow")
                return results
            return []
                
        return self._execute_with_retry(_fetch_articles)
    
    @lru_cache(maxsize=100)
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific knowledge article by ID with caching.
        
        Args:
            article_id (str): The knowledge article ID
            
        Returns:
            Optional[Dict]: The article details or None if not found
        """
        if not self.is_connected():
            logger.error("ServiceNow client not initialized")
            return None
            
        def _fetch_article():
            if self.client:
                kb = self.client.resource(api_path='/table/kb_knowledge')
                response = kb.get(query={'sys_id': article_id})
                result = response.first()
                if result:
                    logger.info(f"Retrieved ServiceNow article: {article_id}")
                    return result
                else:
                    logger.warning(f"ServiceNow article not found: {article_id}")
            return None
                
        return self._execute_with_retry(_fetch_article)
    
    @lru_cache(maxsize=50)
    def search_knowledge_base(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for articles matching the search term with caching.
        
        Args:
            search_term (str): Text to search for
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of matching articles
        """
        if not self.is_connected():
            logger.error("ServiceNow client not initialized")
            return []
            
        def _search_kb():
            if self.client:
                kb = self.client.resource(api_path='/table/kb_knowledge')
                query = {
                    'workflow_state': 'published',
                    'active': 'true',
                    'short_description': f'LIKE{search_term}^ORtext:LIKE{search_term}'
                }
                response = kb.get(query=query, limit=limit)
                results = list(response.all())
                logger.info(f"Found {len(results)} ServiceNow articles matching '{search_term}'")
                return results
            return []
                
        return self._execute_with_retry(_search_kb)
    
    def batch_get_incidents(self, query_list: List[Dict[str, Any]], limit_per_query: int = 50) -> List[Dict[str, Any]]:
        """
        Batch retrieve incidents based on multiple queries.
        
        Args:
            query_list (List[Dict]): List of query parameters
            limit_per_query (int): Maximum number of incidents per query
            
        Returns:
            List[Dict]: Combined list of incidents
        """
        results = []
        for query in query_list:
            query_str = json.dumps(query) if query else ""
            batch_results = self.get_incidents(query_str=query_str, limit=limit_per_query)
            results.extend(batch_results)
        return results
    
    def batch_get_articles(self, query_list: List[Dict[str, Any]], limit_per_query: int = 50) -> List[Dict[str, Any]]:
        """
        Batch retrieve knowledge articles based on multiple queries.
        
        Args:
            query_list (List[Dict]): List of query parameters
            limit_per_query (int): Maximum number of articles per query
            
        Returns:
            List[Dict]: Combined list of articles
        """
        results = []
        for query in query_list:
            query_str = json.dumps(query) if query else ""
            batch_results = self.get_knowledge_articles(query_str=query_str, limit=limit_per_query)
            results.extend(batch_results)
        return results
    
    def export_article_to_json(self, article: Dict[str, Any], output_path: str) -> Optional[str]:
        """
        Export a knowledge article to a JSON file.
        
        Args:
            article (Dict): The article to export
            output_path (str): Path to save the JSON file
            
        Returns:
            Optional[str]: Path to the exported file or None if failed
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(article, f, indent=2)
                
            logger.info(f"Exported ServiceNow article to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error exporting ServiceNow article: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """
        Clear the internal cache for all cached methods.
        """
        self.get_incidents.cache_clear()
        self.get_knowledge_articles.cache_clear()
        self.get_article_by_id.cache_clear()
        self.search_knowledge_base.cache_clear()
        logger.info("ServiceNow connector cache cleared") 