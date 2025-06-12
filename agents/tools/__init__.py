"""
Tools module for agent system
"""

from .visualization_tools import display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files
from .rag_tools import (
    search_pdf_for_streamlit, search_pdf_with_context_for_streamlit, 
    search_pdf_interactive, smart_pdf_search_for_streamlit, display_source_passages,
    get_citation_help, diagnose_pdf_context, validate_source_display_before_final_answer,
    unified_pdf_search_and_analyze
)
from .context_access_tools import (
    check_context_availability, demonstrate_context_access, validate_context_structure
)

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
    
    # RAG tools (Streamlit optimized)
    'search_pdf_for_streamlit',
    'search_pdf_with_context_for_streamlit',
    'search_pdf_interactive', 
    'smart_pdf_search_for_streamlit', 
    'display_source_passages',  # MANDATORY: Display in Streamlit UI
    'get_citation_help',
    'diagnose_pdf_context',
    'validate_source_display_before_final_answer',
    'unified_pdf_search_and_analyze',  # Simplified unified tool
    
    # Context access tools 
    'check_context_availability',
    'demonstrate_context_access',
    'validate_context_structure',
    
    # Web tools
    'enhanced_visit_webpage',
    'bulk_visit_webpages', 
    'extract_financial_data'
]

# Add web tools if available
if enhanced_visit_webpage is not None:
    __all__.extend([
        'enhanced_visit_webpage',
        'bulk_visit_webpages',
        'extract_financial_data'
    ]) 