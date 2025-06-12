"""
Tools module for agent system
"""

from .visualization_tools import display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files
from .rag_tools import unified_pdf_search_and_analyze

try:
    from .enhanced_web_tools import enhanced_visit_webpage, bulk_visit_webpages, extract_financial_data
except ImportError:
    # If enhanced_web_tools doesn't exist, provide fallback
    print("Warning: enhanced_web_tools not found, skipping web tools import")
    enhanced_visit_webpage = None
    bulk_visit_webpages = None
    extract_financial_data = None

__all__ = [
    # Visualization and data tools
    'display_matplotlib_figures',
    'display_plotly_figures', 
    'load_csv_data',
    'discover_data_files',
    
    # RAG tools (unified tool only)
    'unified_pdf_search_and_analyze'
]

# Add web tools if available
if enhanced_visit_webpage is not None:
    __all__.extend([
        'enhanced_visit_webpage',
        'bulk_visit_webpages',
        'extract_financial_data'
    ]) 