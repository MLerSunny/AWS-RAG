"""
ServiceNow connector for retrieving knowledge articles and other data from ServiceNow instances.
"""
import requests
import base64
import json
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ServiceNowConnector:
    def __init__(self, instance_url, username, password):
        """
        Initialize ServiceNow connector with authentication details.
        
        Args:
            instance_url (str): The ServiceNow instance URL
            username (str): ServiceNow username
            password (str): ServiceNow password
        """
        self.instance_url = instance_url
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        logger.info(f"Initialized ServiceNow connector for {instance_url}")
        
    def get_knowledge_articles(self, query="", limit=10):
        """
        Get knowledge articles from ServiceNow.
        
        Args:
            query (str): Optional query string to filter articles
            limit (int): Maximum number of articles to retrieve
            
        Returns:
            list: List of knowledge articles
        """
        try:
            url = f"{self.instance_url}/api/now/table/kb_knowledge"
            headers = {
                "Authorization": f"Basic {self.auth}",
                "Content-Type": "application/json"
            }
            params = {
                "sysparm_query": query,
                "sysparm_limit": limit
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            articles = response.json().get('result', [])
            logger.info(f"Retrieved {len(articles)} knowledge articles from ServiceNow")
            return articles
        except Exception as e:
            logger.error(f"Error retrieving knowledge articles: {str(e)}")
            raise
    
    def get_article_content(self, article_id):
        """
        Get the content of a specific knowledge article.
        
        Args:
            article_id (str): ServiceNow sys_id of the knowledge article
            
        Returns:
            str: Article text content
        """
        try:
            url = f"{self.instance_url}/api/now/table/kb_knowledge/{article_id}"
            headers = {
                "Authorization": f"Basic {self.auth}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            article = response.json().get('result', {})
            content = article.get('text', '')
            
            logger.info(f"Retrieved content for article {article_id}")
            return content
        except Exception as e:
            logger.error(f"Error retrieving article content for {article_id}: {str(e)}")
            raise 