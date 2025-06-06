"""
Enhanced web tools for smolagents with improved content extraction and caching.
"""

from smolagents import tool
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Union
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


class WebContentCache:
    """Simple file-based cache for web content to avoid re-fetching."""
    
    def __init__(self, cache_dir: str = "data/web_cache", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for URL."""
        key = self._get_cache_key(url)
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, url: str) -> Optional[Dict]:
        """Get cached content if still valid."""
        cache_path = self._get_cache_path(url)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                os.remove(cache_path)
                return None
            
            return data['content']
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def set(self, url: str, content: Dict) -> None:
        """Cache content for URL."""
        cache_path = self._get_cache_path(url)
        data = {
            'timestamp': datetime.now().isoformat(),
            'content': content
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to cache content for {url}: {e}")


# Global cache instance
_web_cache = WebContentCache()


@tool
def enhanced_visit_webpage(url: str, extract_links: bool = False, max_content_length: int = 8000) -> str:
    """
    Enhanced webpage visitor with intelligent content extraction, caching, and optimization.
    
    Args:
        url: The URL to visit
        extract_links: Whether to extract relevant links from the page
        max_content_length: Maximum content length to return (prevents token overflow)
        
    Returns:
        Extracted and cleaned content from the webpage
    """
    # Check cache first
    cached = _web_cache.get(url)
    if cached:
        print(f"📋 Using cached content for {url}")
        return _format_webpage_response(cached, extract_links, max_content_length)
    
    try:
        print(f"🌐 Fetching content from: {url}")
        
        # Enhanced headers for better success rate
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Request with timeout and retries
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        
        # Parse with BeautifulSoup for better content extraction
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside", "header"]):
            script.decompose()
        
        # Extract structured content
        content_data = {
            'url': url,
            'title': _extract_title(soup),
            'main_content': _extract_main_content(soup),
            'meta_description': _extract_meta_description(soup),
            'links': _extract_relevant_links(soup, url) if extract_links else [],
            'text_length': len(soup.get_text()),
            'fetch_time': datetime.now().isoformat()
        }
        
        # Cache the content
        _web_cache.set(url, content_data)
        
        return _format_webpage_response(content_data, extract_links, max_content_length)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Failed to fetch {url}: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error processing {url}: {str(e)}"
        print(error_msg)
        return error_msg


@tool 
def bulk_visit_webpages(urls: List[str], max_workers: int = 3, max_content_per_page: int = 5000) -> str:
    """
    Visit multiple webpages concurrently for faster research.
    
    Args:
        urls: List of URLs to visit
        max_workers: Maximum number of concurrent requests
        max_content_per_page: Maximum content length per page
        
    Returns:
        Combined content from all successfully visited pages
    """
    if len(urls) > 10:
        return "❌ Too many URLs provided. Maximum 10 URLs allowed for bulk processing."
    
    print(f"🔄 Bulk fetching {len(urls)} webpages...")
    results = []
    
    def fetch_single_url(url):
        try:
            return enhanced_visit_webpage(url, extract_links=False, max_content_length=max_content_per_page)
        except Exception as e:
            return f"❌ Failed to fetch {url}: {str(e)}"
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single_url, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(f"\n--- Content from {url} ---\n{result}")
            except Exception as e:
                results.append(f"\n--- Error fetching {url} ---\n❌ {str(e)}")
    
    combined_result = "\n".join(results)
    print(f"✅ Bulk fetch completed. Total content length: {len(combined_result)} characters")
    
    return combined_result


@tool
def extract_financial_data(url: str, data_types: List[str] = None) -> str:
    """
    Specialized tool for extracting financial data from webpages.
    
    Args:
        url: URL of the financial page
        data_types: Types of data to extract ['revenue', 'earnings', 'growth', 'margins']
        
    Returns:
        Structured financial data extraction
    """
    if data_types is None:
        data_types = ['revenue', 'earnings', 'growth', 'margins']
    
    content = enhanced_visit_webpage(url, extract_links=False, max_content_length=10000)
    
    # Financial patterns for extraction
    patterns = {
        'revenue': [
            r'revenue[s]?\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)',
            r'net\s*sales[s]?\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)',
            r'total\s*revenue[s]?\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)'
        ],
        'earnings': [
            r'net\s*income\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)',
            r'earnings\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)',
            r'profit\s*(?:of\s*)?[\$]?([0-9,]+\.?[0-9]*)\s*(?:billion|million|B|M)'
        ],
        'growth': [
            r'growth\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'increase\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'up\s*([0-9]+\.?[0-9]*%?)'
        ],
        'margins': [
            r'margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'operating\s*margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'profit\s*margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)'
        ]
    }
    
    extracted_data = {}
    
    for data_type in data_types:
        if data_type in patterns:
            matches = []
            for pattern in patterns[data_type]:
                found = re.findall(pattern, content, re.IGNORECASE)
                matches.extend(found)
            extracted_data[data_type] = list(set(matches))  # Remove duplicates
    
    # Format results
    result = f"🏦 Financial Data Extracted from {url}:\n\n"
    for data_type, values in extracted_data.items():
        if values:
            result += f"📊 {data_type.title()}: {', '.join(values[:5])}\n"  # Limit to 5 values
        else:
            result += f"📊 {data_type.title()}: No data found\n"
    
    result += f"\n📄 Full Content Preview:\n{content[:1000]}..."
    
    return result


# Helper functions
def _extract_title(soup: BeautifulSoup) -> str:
    """Extract page title."""
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # Fallback to h1
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    return "No title found"


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content using common content selectors."""
    # Try common content selectors
    content_selectors = [
        'main', 'article', '[role="main"]', '.content', '.main-content',
        '.post-content', '.entry-content', '.article-content', '.story-content'
    ]
    
    for selector in content_selectors:
        content_elem = soup.select_one(selector)
        if content_elem:
            return _clean_text(content_elem.get_text())
    
    # Fallback to body text
    body = soup.find('body')
    if body:
        return _clean_text(body.get_text())
    
    return _clean_text(soup.get_text())


def _extract_meta_description(soup: BeautifulSoup) -> str:
    """Extract meta description."""
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        return meta_desc.get('content', '').strip()
    
    # Try Open Graph description
    og_desc = soup.find('meta', property='og:description')
    if og_desc:
        return og_desc.get('content', '').strip()
    
    return ""


def _extract_relevant_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Extract relevant links from the page."""
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text().strip()
        
        # Skip empty links or anchors
        if not href or href.startswith('#') or not text:
            continue
        
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        
        # Skip same-domain duplicates and common irrelevant links
        if _is_relevant_link(text, full_url):
            links.append({
                'url': full_url,
                'text': text[:100],  # Limit text length
                'domain': urlparse(full_url).netloc
            })
    
    return links[:10]  # Limit to 10 most relevant links


def _is_relevant_link(text: str, url: str) -> bool:
    """Check if a link is relevant for content extraction."""
    # Skip common navigation/footer links
    skip_terms = [
        'privacy', 'terms', 'cookie', 'contact', 'about us', 'login',
        'register', 'subscribe', 'newsletter', 'follow us', 'facebook',
        'twitter', 'linkedin', 'instagram', 'youtube'
    ]
    
    text_lower = text.lower()
    url_lower = url.lower()
    
    # Skip if contains skip terms
    if any(term in text_lower or term in url_lower for term in skip_terms):
        return False
    
    # Keep if text is meaningful (not just "read more", "click here", etc.)
    if len(text.strip()) < 5:
        return False
    
    return True


def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common footer/header noise
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Skip very short lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _format_webpage_response(content_data: Dict, extract_links: bool, max_length: int) -> str:
    """Format the webpage response for agent consumption."""
    response_parts = []
    
    # Title and meta
    if content_data.get('title'):
        response_parts.append(f"📄 Title: {content_data['title']}")
    
    if content_data.get('meta_description'):
        response_parts.append(f"📝 Description: {content_data['meta_description']}")
    
    # Main content
    main_content = content_data.get('main_content', '')
    if len(main_content) > max_length:
        main_content = main_content[:max_length] + "... [Content truncated]"
    
    response_parts.append(f"📖 Content:\n{main_content}")
    
    # Links if requested
    if extract_links and content_data.get('links'):
        links_text = "\n".join([f"  • {link['text']}: {link['url']}" 
                               for link in content_data['links'][:5]])
        response_parts.append(f"🔗 Relevant Links:\n{links_text}")
    
    return "\n\n".join(response_parts) 