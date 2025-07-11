"""
Web search and content extraction functionality.
"""

import time
import backoff
import os
from typing import List, Set, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import concurrent.futures

from langchain_community.document_transformers import Html2TextTransformer

from utils.logger import logger
from common.config import DEFAULT_MAX_RESULTS
from common.config import GOOGLE_API_KEY, GOOGLE_CSE_ID, MAX_WORKERS

class SearchExtractor:
    """
    Component for performing web searches and extracting content from web pages.
    """
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        """
        Initialize the search extractor.
        
        Args:
            max_workers: Maximum number of parallel search workers
        """
        self.max_workers = max_workers
        self.session = self._create_session()
        self.html2text = Html2TextTransformer()

    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with appropriate headers.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })
        return session
    
    def _perform_google_search(self, query: str, num_results: int = 10) -> List[str]:
        """
        Perform a search using Google Custom Search API.
        
        Args:
            query: Search query
            num_results: Number of search results to return
            
        Returns:
            List of search result URLs
        """
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            logger.warning("Google Custom Search API credentials not configured")
            return []
            
        try:
            # Configure Google Custom Search API parameters
            params = {
                "q": query,
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "num": min(num_results, 10)  # Google CSE limits to 10 results per request
            }
            
            # Execute the search
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            results = response.json()
            
            # Extract URLs from results
            urls = []
            if "items" in results:
                urls = [item.get("link") for item in results["items"] 
                       if item.get("link") and self._is_valid_url(item.get("link"))]
                
            # logger.info(f"Google Custom Search found {len(urls)} results for query '{query}'")
            return urls
            
        except Exception as e:
            logger.error(f"Error during Google Custom Search for query '{query}': {str(e)}")
            return []
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and not a file or other non-web resource.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme) and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def _perform_single_search(self, query: str, num_results: int = 10) -> List[str]:
        """
        Perform a single web search.
        
        Args:
            query: Search query
            num_results: Number of search results to return
            
        Returns:
            List of search result URLs
        """
        search_results = []
        
        # Try Google Custom Search
        search_results = self._perform_google_search(query, num_results)
        if search_results:
            return search_results
            
        logger.warning(f"No search results found for query '{query}'")
        return []
    
    def search(self, query: str, num_results: int = DEFAULT_MAX_RESULTS) -> List[str]:
        """
        Perform a web search with retry logic.
        
        Args:
            query: Search query
            num_results: Number of search results to return
            
        Returns:
            List of search result URLs
        """
        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        def _search_with_retry():
            return self._perform_single_search(query, num_results)
            
        return _search_with_retry()
    
    def multi_query_search(self, queries: List[str], results_per_query: int = 3) -> List[str]:
        """
        Perform multiple searches in parallel and combine results.
        
        Args:
            queries: List of search queries
            results_per_query: Number of results to get per query
            
        Returns:
            List of unique search result URLs
        """
        all_urls = set()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(self.search, query, results_per_query): query 
                for query in queries
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    urls = future.result()
                    all_urls.update(urls)
                except Exception as e:
                    logger.error(f"Error searching for query '{query}': {str(e)}")
        
        # Convert set back to list
        return list(all_urls)
    
    def extract_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract content from a list of URLs.
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            List of dictionaries containing extracted content
        """
        results = []
        
        def process_url(url: str) -> Dict[str, Any]:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else ""
                
                # Extract main content
                main_content = ""
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    main_content += tag.get_text() + "\n"
                
                return {
                    "url": url,
                    "title": title,
                    "content": main_content.strip()
                }
                
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {str(e)}")
                return {
                    "url": url,
                    "title": "",
                    "content": ""
                }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_url, urls))
            
        return results 