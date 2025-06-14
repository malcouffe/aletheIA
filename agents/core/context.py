"""
Context Management for Multi-Agent Systems - Enhanced with Debug
Handles context preparation and passing between agents following smolagents best practices.
"""

import logging
from typing import List, Dict, Any, Optional

# Configure logging for context management debug - disable HTTP/API logs
logging.basicConfig(level=logging.WARNING)  # Set global level to WARNING
logger = logging.getLogger(__name__)

# Disable verbose HTTP/API logs but keep RAG debug prints
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class ContextManager:
    """Manages context data for multi-agent conversations with debug logging."""
    
    def __init__(self):
        self.pdf_context: Optional[Dict] = None
        self.csv_context: Optional[Dict] = None
        print("🏗️ Context Manager: Initialized new context manager")
    
    def set_pdf_context(self, available_pdfs: List[Dict]):
        """Set PDF context from available PDF files with debug logging."""
        print(f"📄 Context Manager: Setting PDF context with {len(available_pdfs)} files")
        
        if available_pdfs:
            self.pdf_context = {
                "available_files": available_pdfs,
                "count": len(available_pdfs),
                "classifications": list(set(
                    pdf.get('classification', 'General') 
                    for pdf in available_pdfs
                ))
            }
            
            print(f"✅ Context Manager: PDF context set - {len(available_pdfs)} files")
            for i, pdf in enumerate(available_pdfs, 1):
                filename = pdf.get('filename', 'Unknown')
                classification = pdf.get('classification', 'General')
                print(f"📄 Context Manager: PDF {i}: {filename} ({classification})")
        else:
            print("⚠️ Context Manager: No PDF files provided")
    
    def set_csv_context(self, available_csvs: List[Dict]):
        """Set CSV context from available CSV files with debug logging."""
        print(f"📊 Context Manager: Setting CSV context with {len(available_csvs)} files")
        
        if available_csvs:
            self.csv_context = {
                "available_files": available_csvs,
                "count": len(available_csvs),
                "total_columns": sum(
                    len(csv.get('csv_args', {}).get('columns', [])) 
                    for csv in available_csvs
                )
            }
            
            print(f"✅ Context Manager: CSV context set - {len(available_csvs)} files")
            total_columns = self.csv_context["total_columns"]
            print(f"📊 Context Manager: Total columns across all CSVs: {total_columns}")
            
            for i, csv in enumerate(available_csvs, 1):
                filename = csv.get('filename', 'Unknown')
                columns = len(csv.get('csv_args', {}).get('columns', []))
                print(f"📊 Context Manager: CSV {i}: {filename} ({columns} columns)")
        else:
            print("⚠️ Context Manager: No CSV files provided")
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get the complete context dictionary for agent additional_args with debug logging."""
        print("🔍 Context Manager: Building context dictionary for additional_args")
        
        context = {}
        
        if self.pdf_context:
            context["pdf_context"] = self.pdf_context
            print(f"📄 Context Manager: Added PDF context ({self.pdf_context['count']} files)")
        
        if self.csv_context:
            context["csv_context"] = self.csv_context
            print(f"📊 Context Manager: Added CSV context ({self.csv_context['count']} files)")
        
        print(f"✅ Context Manager: Context dictionary built with {len(context)} contexts")
        return context
    
    def get_context_summary(self) -> List[str]:
        """Get human-readable context summary for task descriptions with debug logging."""
        print("📝 Context Manager: Generating context summary")
        
        context_hints = []
        
        if self.pdf_context:
            pdf_count = self.pdf_context["count"]
            classifications = self.pdf_context["classifications"]
            hint = f"PDF Documents: {pdf_count} files available ({', '.join(classifications)} topics)"
            context_hints.append(hint)
            print(f"📄 Context Manager: Added PDF hint: {hint}")
        
        if self.csv_context:
            csv_count = self.csv_context["count"]
            hint = f"CSV Datasets: {csv_count} files available for analysis"
            context_hints.append(hint)
            print(f"📊 Context Manager: Added CSV hint: {hint}")
        
        if not context_hints:
            hint = "Web search capabilities"
            context_hints.append(hint)
            print(f"🌐 Context Manager: Default hint: {hint}")
        
        print(f"✅ Context Manager: Generated {len(context_hints)} context hints")
        return context_hints
    



def prepare_manager_context(
    available_pdfs_context: Optional[List[Dict]] = None,
    available_csvs_context: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Prepare context data for the manager agent using additional_args.
    Enhanced with debug logging following smolagents best practices.
    
    Args:
        available_pdfs_context: List of PDF file context dictionaries
        available_csvs_context: List of CSV file context dictionaries
    
    Returns:
        Context dictionary for additional_args
    """
    print("🏗️ Context Prep: Preparing manager context")
    print(f"📄 Context Prep: PDFs provided: {len(available_pdfs_context) if available_pdfs_context else 0}")
    print(f"📊 Context Prep: CSVs provided: {len(available_csvs_context) if available_csvs_context else 0}")
    
    manager = ContextManager()
    
    if available_pdfs_context:
        print("📄 Context Prep: Setting PDF context")
        manager.set_pdf_context(available_pdfs_context)
    
    if available_csvs_context:
        print("📊 Context Prep: Setting CSV context")
        manager.set_csv_context(available_csvs_context)
    
    context = manager.get_context_dict()
    
    print(f"✅ Context Prep: Manager context prepared with {len(context)} context types")
    return context


def build_simple_manager_task(
    user_query: str,
    available_pdfs_context: Optional[List[Dict]] = None,
    available_csvs_context: Optional[List[Dict]] = None
) -> str:
    """
    Build a simple task description for the manager agent.
    Enhanced with debug logging and smolagents best practices.
    
    Args:
        user_query: The user's query/request (already enriched by contextual agent)
        available_pdfs_context: List of PDF file context dictionaries  
        available_csvs_context: List of CSV file context dictionaries
    
    Returns:
        Formatted task description optimized for smolagents delegation
    """
    print("📝 Task Builder: Building manager task description")
    print(f"❓ Task Builder: Enriched query: '{user_query}'")
    
    manager = ContextManager()
    
    if available_pdfs_context:
        print("📄 Task Builder: Including PDF context in task")
        manager.set_pdf_context(available_pdfs_context)
    
    if available_csvs_context:
        print("📊 Task Builder: Including CSV context in task")
        manager.set_csv_context(available_csvs_context)
    
    context_hints = manager.get_context_summary()
    print(f"💡 Task Builder: Generated {len(context_hints)} context hints")
    
    # Enhanced task format for better smolagents delegation
    task = f"""Enriched Query: "{user_query}"

Available Resources:
{chr(10).join(f"- {hint}" for hint in context_hints)}

Instructions for Manager: 
- Analyze the enriched query and available resources
- Delegate IMMEDIATELY to the appropriate specialist agent
- Use debug logging to show delegation decision
- Pass the complete enriched query to the specialist
"""
    print("✅ Task Builder: Task description built successfully")
    return task


 