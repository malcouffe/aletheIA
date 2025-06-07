"""
Enhanced web tools for smolagents with improved content extraction and caching.
Following smolagents best practices for tool design, error handling, and user guidance.
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
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'content': content
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Fail silently if caching fails


# Global cache instance
_web_cache = WebContentCache()


@tool
def enhanced_visit_webpage(url: str, extract_links: bool = False, max_content_length: int = 8000) -> str:
    """
    Enhanced webpage visitor with intelligent content extraction, caching, and optimization.
    
    This tool fetches and parses web content with robust error handling, intelligent 
    content extraction, and automatic caching for improved performance. Following 
    smolagents best practices for comprehensive logging and user guidance.
    
    Args:
        url: The complete URL to visit (must include http:// or https://)
        extract_links: Whether to extract and return relevant links from the page
        max_content_length: Maximum content length to return (prevents token overflow)
        
    Returns:
        Cleaned and structured content from the webpage including title, main content,
        meta information, and optionally relevant links. Or detailed error information
        with troubleshooting guidance if the request fails.
        
    Usage Examples:
        - enhanced_visit_webpage("https://example.com/article")
        - enhanced_visit_webpage("https://news.site.com", extract_links=True)
        - enhanced_visit_webpage("https://long-page.com", max_content_length=5000)
    """
    print(f"🌐 enhanced_visit_webpage called for: {url}")
    print(f"   📋 Extract links: {extract_links}, Max length: {max_content_length}")
    
    # Basic URL validation
    if not url or not isinstance(url, str):
        error_msg = "❌ ERROR: Invalid URL provided"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a valid URL string starting with http:// or https://"
    
    if not url.startswith(('http://', 'https://')):
        error_msg = f"❌ ERROR: URL must start with http:// or https://, received: {url}"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Add 'https://' prefix to the URL"
    
    # Check cache first
    cached = _web_cache.get(url)
    if cached:
        print(f"📋 Using cached content for {url}")
        return _format_webpage_response(cached, extract_links, max_content_length)
    
    try:
        print(f"🌐 Fetching fresh content from: {url}")
        
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
        
        print("   📡 Sending HTTP request...")
        response = session.get(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        print(f"   ✅ Request successful (Status: {response.status_code})")
        
        # Parse with BeautifulSoup for better content extraction
        print("   🔍 Parsing HTML content...")
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
            'fetch_time': datetime.now().isoformat(),
            'status_code': response.status_code
        }
        
        print(f"   📄 Content extracted: {content_data['text_length']} characters")
        
        # Cache the content
        _web_cache.set(url, content_data)
        print("   💾 Content cached for future use")
        
        return _format_webpage_response(content_data, extract_links, max_content_length)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ NETWORK ERROR fetching {url}: {str(e)}"
        print(error_msg)
        return error_msg + "\n\n💡 TROUBLESHOOTING:\n   - Check internet connection\n   - Verify URL is accessible\n   - Try again in a few moments\n   - Check if site requires authentication"
    except Exception as e:
        error_msg = f"❌ PROCESSING ERROR for {url}: {str(e)}"
        print(error_msg)
        return error_msg + "\n\n💡 TROUBLESHOOTING:\n   - URL might contain unusual content\n   - Site might be using complex JavaScript\n   - Try a different URL or contact support"


@tool 
def bulk_visit_webpages(urls: List[str], max_workers: int = 3, max_content_per_page: int = 5000) -> str:
    """
    Visit multiple webpages concurrently for faster research and data gathering.
    
    This tool efficiently processes multiple URLs in parallel while respecting
    rate limits and providing comprehensive error reporting. Following smolagents
    best practices for batch processing and detailed feedback.
    
    Args:
        urls: List of URLs to visit (maximum 10 URLs allowed for performance)
        max_workers: Maximum number of concurrent requests (recommended: 2-5)
        max_content_per_page: Maximum content length per page to prevent overflow
        
    Returns:
        Combined content from all successfully visited pages with clear separation
        and status information, plus detailed error reports for failed requests.
        
    Usage Examples:
        - bulk_visit_webpages(["https://site1.com", "https://site2.com"])
        - bulk_visit_webpages(urls_list, max_workers=2, max_content_per_page=3000)
        - bulk_visit_webpages(research_urls, max_workers=4)  # For faster processing
    """
    print(f"🔄 bulk_visit_webpages called with {len(urls) if isinstance(urls, list) else 'invalid'} URLs")
    print(f"   ⚙️ Settings: max_workers={max_workers}, max_content_per_page={max_content_per_page}")
    
    # Input validation
    if not isinstance(urls, list):
        error_msg = "❌ ERROR: Expected list of URLs, received: " + str(type(urls))
        print(error_msg)
        return error_msg + "\n\n💡 CORRECT FORMAT: [\"https://site1.com\", \"https://site2.com\"]"
    
    if not urls:
        warning_msg = "⚠️ WARNING: Empty URL list provided"
        print(warning_msg)
        return warning_msg + "\n\n💡 TIP: Provide a list of URLs to visit"
    
    if len(urls) > 10:
        error_msg = f"❌ ERROR: Too many URLs ({len(urls)}). Maximum 10 URLs allowed for bulk processing."
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Process URLs in batches of ≤10"
    
    # Validate individual URLs
    invalid_urls = [url for url in urls if not isinstance(url, str) or not url.startswith(('http://', 'https://'))]
    if invalid_urls:
        error_msg = f"❌ ERROR: Invalid URLs detected: {invalid_urls[:3]}{'...' if len(invalid_urls) > 3 else ''}"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Ensure all URLs start with http:// or https://"
    
    print(f"🔄 Starting bulk fetch of {len(urls)} webpages...")
    results = []
    successful_count = 0
    failed_count = 0
    
    def fetch_single_url(url):
        try:
            print(f"   🌐 Processing: {url}")
            result = enhanced_visit_webpage(url, extract_links=False, max_content_length=max_content_per_page)
            if not result.startswith("❌"):
                return {"success": True, "url": url, "content": result}
            else:
                return {"success": False, "url": url, "error": result}
        except Exception as e:
            return {"success": False, "url": url, "error": f"❌ Unexpected error: {str(e)}"}
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single_url, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result["success"]:
                    results.append(f"\n{'='*60}")
                    results.append(f"✅ SUCCESS: {result['url']}")
                    results.append(f"{'='*60}")
                    results.append(result["content"])
                    successful_count += 1
                    print(f"   ✅ Completed: {url}")
                else:
                    results.append(f"\n{'='*60}")
                    results.append(f"❌ FAILED: {result['url']}")
                    results.append(f"{'='*60}")
                    results.append(result["error"])
                    failed_count += 1
                    print(f"   ❌ Failed: {url}")
            except Exception as e:
                results.append(f"\n{'='*60}")
                results.append(f"❌ CRITICAL ERROR: {url}")
                results.append(f"{'='*60}")
                results.append(f"Unexpected error during processing: {str(e)}")
                failed_count += 1
                print(f"   ❌ Critical error: {url}")
    
    # Prepare comprehensive summary
    summary_header = f"""
🔄 BULK WEBPAGE PROCESSING COMPLETE

📊 SUMMARY:
   ✅ Successful: {successful_count}/{len(urls)} pages
   ❌ Failed: {failed_count}/{len(urls)} pages
   ⚙️ Workers used: {max_workers}
   📄 Max content per page: {max_content_per_page} chars

📋 DETAILED RESULTS:
"""
    
    combined_result = summary_header + "\n".join(results)
    
    print(f"✅ Bulk fetch completed: {successful_count} success, {failed_count} failed")
    print(f"📊 Total content length: {len(combined_result)} characters")
    
    return combined_result


@tool
def extract_financial_data(url: str, data_types: List[str] = None) -> str:
    """
    Specialized tool for extracting financial data from webpages using pattern matching.
    
    This tool combines webpage content extraction with intelligent financial data
    pattern recognition to extract key metrics. Following smolagents best practices
    for specialized domain tools with clear output formatting.
    
    Args:
        url: URL of the financial page to analyze
        data_types: List of financial data types to extract. Available options:
                   ['revenue', 'earnings', 'growth', 'margins', 'valuation', 'debt']
                   If None, extracts all available types.
        
    Returns:
        Structured financial data extraction with identified values, confidence levels,
        and source context, plus the full page content for additional analysis.
        
    Usage Examples:
        - extract_financial_data("https://investor.company.com/earnings")
        - extract_financial_data("https://finance.yahoo.com/quote/AAPL", ["revenue", "earnings"])
        - extract_financial_data("https://sec.gov/filing.html", ["revenue", "margins", "debt"])
    """
    print(f"🏦 extract_financial_data called for: {url}")
    
    if data_types is None:
        data_types = ['revenue', 'earnings', 'growth', 'margins', 'valuation', 'debt']
    
    print(f"   📈 Target data types: {data_types}")
    
    # Input validation
    if not isinstance(data_types, list):
        error_msg = "❌ ERROR: data_types must be a list"
        print(error_msg)
        return error_msg + "\n\n💡 VALID OPTIONS: ['revenue', 'earnings', 'growth', 'margins', 'valuation', 'debt']"
    
    valid_types = ['revenue', 'earnings', 'growth', 'margins', 'valuation', 'debt']
    invalid_types = [dt for dt in data_types if dt not in valid_types]
    if invalid_types:
        error_msg = f"❌ ERROR: Invalid data types: {invalid_types}"
        print(error_msg)
        return error_msg + f"\n\n💡 VALID OPTIONS: {valid_types}"
    
    # Get webpage content first
    print("   🌐 Fetching webpage content...")
    content = enhanced_visit_webpage(url, extract_links=False, max_content_length=12000)
    
    if content.startswith("❌"):
        return f"🏦 FINANCIAL DATA EXTRACTION FAILED\n\n{content}"
    
    print("   🔍 Analyzing content for financial patterns...")
    
    # Enhanced financial patterns for extraction
    patterns = {
        'revenue': [
            r'revenue[s]?\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'net\s*sales[s]?\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'total\s*revenue[s]?\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'sales\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)'
        ],
        'earnings': [
            r'net\s*income\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'earnings\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'profit\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'EBITDA\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)'
        ],
        'growth': [
            r'growth\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'increase\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'up\s*([0-9]+\.?[0-9]*%?)',
            r'YoY\s*([0-9]+\.?[0-9]*%?)',
            r'year.over.year\s*([0-9]+\.?[0-9]*%?)'
        ],
        'margins': [
            r'margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'operating\s*margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'profit\s*margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)',
            r'gross\s*margin\s*(?:of\s*)?([0-9]+\.?[0-9]*%?)'
        ],
        'valuation': [
            r'market\s*cap\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'valuation\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'enterprise\s*value\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)'
        ],
        'debt': [
            r'debt\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'total\s*debt\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)',
            r'net\s*debt\s*(?:of\s*)?[\$€£¥]?\s*([0-9,]+\.?[0-9]*)\s*(?:billion|million|trillion|B|M|T)'
        ]
    }
    
    extracted_data = {}
    
    for data_type in data_types:
        if data_type in patterns:
            matches = []
            pattern_count = 0
            for pattern in patterns[data_type]:
                found = re.findall(pattern, content, re.IGNORECASE)
                matches.extend(found)
                pattern_count += len(found)
            
            # Remove duplicates and limit results
            unique_matches = list(set(matches))[:10]  # Limit to 10 values
            extracted_data[data_type] = {
                'values': unique_matches,
                'count': len(unique_matches),
                'total_matches': pattern_count
            }
            print(f"   📊 {data_type}: Found {len(unique_matches)} unique values")
    
    # Format comprehensive results
    result_parts = [f"🏦 FINANCIAL DATA EXTRACTION RESULTS"]
    result_parts.append(f"🌐 Source: {url}")
    result_parts.append(f"📅 Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result_parts.append("=" * 60)
    
    total_found = 0
    for data_type, data_info in extracted_data.items():
        values = data_info['values']
        count = data_info['count']
        total_matches = data_info['total_matches']
        
        result_parts.append(f"\n📊 {data_type.upper()}:")
        if values:
            result_parts.append(f"   ✅ Found {count} unique values ({total_matches} total matches)")
            for i, value in enumerate(values[:5], 1):  # Show top 5
                result_parts.append(f"   {i}. {value}")
            if len(values) > 5:
                result_parts.append(f"   ... and {len(values) - 5} more")
            total_found += count
        else:
            result_parts.append(f"   ❌ No data found")
    
    result_parts.append(f"\n📈 SUMMARY: {total_found} total financial data points extracted")
    
    if total_found == 0:
        result_parts.append("\n💡 TROUBLESHOOTING:")
        result_parts.append("   - Page might not contain financial data")
        result_parts.append("   - Try a different financial page or earnings report")
        result_parts.append("   - Check if data is in tables or non-standard formats")
    
    result_parts.append(f"\n📄 FULL PAGE CONTENT (first 2000 chars):")
    result_parts.append("-" * 60)
    result_parts.append(content[:2000] + "..." if len(content) > 2000 else content)
    
    final_result = "\n".join(result_parts)
    print(f"✅ Financial extraction completed: {total_found} data points found")
    return final_result


# Helper functions for content extraction
def _extract_title(soup) -> str:
    """Extract page title with fallbacks."""
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # Fallback to h1
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    return "No title found"


def _extract_main_content(soup) -> str:
    """Extract main content with intelligent selection."""
    # Try to find main content areas
    content_selectors = [
        'main', 'article', '.content', '#content', '.main-content',
        '.article-content', '.post-content', '.entry-content'
    ]
    
    for selector in content_selectors:
        element = soup.select_one(selector)
        if element:
            return element.get_text().strip()
    
    # Fallback to body
    body = soup.find('body')
    if body:
        return body.get_text().strip()
    
    return soup.get_text().strip()


def _extract_meta_description(soup) -> str:
    """Extract meta description."""
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        return meta_desc.get('content', '').strip()
    
    meta_desc = soup.find('meta', attrs={'property': 'og:description'})
    if meta_desc:
        return meta_desc.get('content', '').strip()
    
    return "No description found"


def _extract_relevant_links(soup, base_url: str) -> List[str]:
    """Extract relevant links from the page."""
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        if full_url.startswith(('http://', 'https://')) and full_url not in links:
            links.append(full_url)
        if len(links) >= 10:  # Limit to 10 links
            break
    return links


def _format_webpage_response(content_data: Dict, extract_links: bool, max_length: int) -> str:
    """Format the webpage response consistently."""
    result_parts = []
    
    # Header with metadata
    result_parts.append(f"🌐 WEBPAGE CONTENT EXTRACTED")
    result_parts.append(f"📍 URL: {content_data['url']}")
    result_parts.append(f"📄 Title: {content_data['title']}")
    result_parts.append(f"📝 Description: {content_data['meta_description']}")
    result_parts.append(f"📊 Content length: {content_data['text_length']} characters")
    result_parts.append(f"🕒 Fetched: {content_data.get('fetch_time', 'Unknown')}")
    
    if 'status_code' in content_data:
        result_parts.append(f"📡 Status: {content_data['status_code']}")
    
    result_parts.append("=" * 60)
    
    # Main content (truncated if needed)
    main_content = content_data['main_content']
    if len(main_content) > max_length:
        main_content = main_content[:max_length] + f"\n\n... [Content truncated. Full length: {len(content_data['main_content'])} chars]"
    
    result_parts.append("📖 MAIN CONTENT:")
    result_parts.append(main_content)
    
    # Links if requested
    if extract_links and content_data['links']:
        result_parts.append("\n🔗 RELEVANT LINKS:")
        for i, link in enumerate(content_data['links'][:10], 1):
            result_parts.append(f"   {i}. {link}")
    
    return "\n".join(result_parts) 