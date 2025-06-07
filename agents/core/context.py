"""
Context Management for Multi-Agent Systems
Handles context preparation and passing between agents.
"""

from typing import List, Dict, Any, Optional


class ContextManager:
    """Manages context data for multi-agent conversations."""
    
    def __init__(self):
        self.pdf_context: Optional[Dict] = None
        self.csv_context: Optional[Dict] = None
    
    def set_pdf_context(self, available_pdfs: List[Dict]):
        """Set PDF context from available PDF files."""
        if available_pdfs:
            self.pdf_context = {
                "available_files": available_pdfs,
                "count": len(available_pdfs),
                "classifications": list(set(
                    pdf.get('classification', 'General') 
                    for pdf in available_pdfs
                ))
            }
    
    def set_csv_context(self, available_csvs: List[Dict]):
        """Set CSV context from available CSV files."""
        if available_csvs:
            self.csv_context = {
                "available_files": available_csvs,
                "count": len(available_csvs),
                "total_columns": sum(
                    len(csv.get('csv_args', {}).get('columns', [])) 
                    for csv in available_csvs
                )
            }
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get the complete context dictionary for agent additional_args."""
        context = {}
        
        if self.pdf_context:
            context["pdf_context"] = self.pdf_context
        
        if self.csv_context:
            context["csv_context"] = self.csv_context
        
        return context
    
    def get_context_summary(self) -> List[str]:
        """Get human-readable context summary for task descriptions."""
        context_hints = []
        
        if self.pdf_context:
            pdf_count = self.pdf_context["count"]
            classifications = self.pdf_context["classifications"]
            context_hints.append(
                f"PDF Documents: {pdf_count} files available "
                f"({', '.join(classifications)} topics)"
            )
        
        if self.csv_context:
            csv_count = self.csv_context["count"]
            context_hints.append(
                f"CSV Datasets: {csv_count} files available for analysis"
            )
        
        if not context_hints:
            context_hints.append("Web search capabilities")
        
        return context_hints
    
    def clear(self):
        """Clear all context data."""
        self.pdf_context = None
        self.csv_context = None


def prepare_manager_context(
    available_pdfs_context: Optional[List[Dict]] = None,
    available_csvs_context: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Prepare context data for the manager agent using additional_args.
    
    Args:
        available_pdfs_context: List of PDF file context dictionaries
        available_csvs_context: List of CSV file context dictionaries
    
    Returns:
        Context dictionary for additional_args
    """
    manager = ContextManager()
    
    if available_pdfs_context:
        manager.set_pdf_context(available_pdfs_context)
    
    if available_csvs_context:
        manager.set_csv_context(available_csvs_context)
    
    context = manager.get_context_dict()
    
    return context


def build_simple_manager_task(
    user_query: str,
    available_pdfs_context: Optional[List[Dict]] = None,
    available_csvs_context: Optional[List[Dict]] = None
) -> str:
    """
    Build a simple task description for the manager agent.
    
    Args:
        user_query: The user's query/request
        available_pdfs_context: List of PDF file context dictionaries  
        available_csvs_context: List of CSV file context dictionaries
    
    Returns:
        Formatted task description
    """
    manager = ContextManager()
    
    if available_pdfs_context:
        manager.set_pdf_context(available_pdfs_context)
    
    if available_csvs_context:
        manager.set_csv_context(available_csvs_context)
    
    context_hints = manager.get_context_summary()
    
    task = f"""User Query: "{user_query}"

Available Resources:
{chr(10).join(f"- {hint}" for hint in context_hints)}

Route to the appropriate agent and provide a natural, conversational response in French."""

    return task 