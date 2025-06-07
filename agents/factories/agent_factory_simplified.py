"""
Simplified Agent Factory for Streamlined Architecture
Creates only Manager Agent (with all tools) and Data Analyst Agent.
"""

import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
from typing import Optional

from ..config.agent_config import AGENT_CONFIGS, AGENT_DESCRIPTIONS
from ..tools import (
    display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files,
    search_pdf_documents, search_pdf_with_context, search_pdf_from_state, get_citation_help
)
from ..tools.context_access_tools import check_context_availability, demonstrate_context_access, validate_context_structure
from ..tools.enhanced_web_tools import enhanced_visit_webpage, bulk_visit_webpages, extract_financial_data
from ..core.embedding import get_embedding_function


class SimplifiedAgentFactory:
    """Simplified factory for creating a streamlined agent system."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.embedding_function = get_embedding_function()
    
    def create_data_analyst_agent(self) -> CodeAgent:
        """Create a data analyst agent with visualization capabilities."""
        config = AGENT_CONFIGS["data_analyst"]
        
        agent = CodeAgent(
            tools=[display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files],
            model=self.model,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib.pyplot", "seaborn", 
                "plotly.express", "plotly.graph_objects", "plotly.subplots", "scipy.stats",
                "sklearn.preprocessing", "sklearn.cluster", "warnings",
                "streamlit", "os", "sys", "glob",
                # Additional useful libraries for comprehensive data analysis
                "datetime", "math", "statistics", "json", "csv",
                "matplotlib.patches", "matplotlib.colors", "pandas.api.types",
                "scipy.cluster", "scipy.optimize", "sklearn.metrics", "sklearn.model_selection"
            ],
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="data_analyst",
            description=AGENT_DESCRIPTIONS["data_analyst"],
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval
        )
        
        print(f"✅ Data analyst agent created successfully")
        return agent

    def create_enhanced_manager_agent(self, data_analyst_agent: CodeAgent) -> CodeAgent:
        """Create an enhanced manager agent with all tools (web, RAG, delegation)."""
        config = AGENT_CONFIGS["manager_agent"]
        
        # Combine all tools: web tools, RAG tools
        all_tools = []
        
        # Add web search tools
        try:
            search_tool = DuckDuckGoSearchTool()
            all_tools.extend([
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages, 
                extract_financial_data
            ])
            print("✅ Web search tools added to manager")
        except ImportError as ie:
            print(f"⚠️ Web search tools unavailable: {ie}")
        
        # Add RAG tools
        try:
            all_tools.extend([
                search_pdf_from_state, 
                search_pdf_with_context, 
                search_pdf_documents,
                get_citation_help
            ])
            print("✅ RAG tools added to manager")
        except Exception as e:
            print(f"⚠️ RAG tools partially unavailable: {e}")
        
        # Add context access tools for debugging and guidance
        try:
            all_tools.extend([
                check_context_availability,
                demonstrate_context_access,
                validate_context_structure
            ])
            print("✅ Context access tools added to manager")
        except Exception as e:
            print(f"⚠️ Context access tools unavailable: {e}")
        
        # Create enhanced manager description
        enhanced_description = """Expert task manager with comprehensive capabilities.

CORE CAPABILITIES:
1. **Direct Web Research**: Enhanced web search and content extraction
2. **Document Analysis**: PDF search and retrieval using RAG
3. **Data Analysis Delegation**: Delegate complex data tasks to data_analyst

AVAILABLE TOOLS:
- Web: DuckDuckGoSearchTool, enhanced_visit_webpage, bulk_visit_webpages, extract_financial_data
- RAG: search_pdf_from_state, search_pdf_with_context, search_pdf_documents, get_citation_help
- Context: check_context_availability, demonstrate_context_access, validate_context_structure
- Delegation: data_analyst agent for CSV analysis and visualizations

CRITICAL DELEGATION RULES:
1. **ALWAYS DELEGATE DATA ANALYSIS** → to data_analyst:
   - Keywords: "analyser", "dataset", "CSV", "données", "visualisation", "graphique", "statistiques", "tableau", "colonnes", "lignes", "tendance", "corrélation", "moyenne", "médiane", "distribution"
   - **MANDATORY FIRST STEP**: Check for CSV context and delegate immediately
   - Pattern:
     ```python
     # Check for CSV context first
     print("🔍 Checking for CSV context...")
     csv_available = 'csv_context' in globals()
     print(f"CSV context available: {csv_available}")
     
     if csv_available:
         csv_context = globals()['csv_context']
         print(f"📊 Found {csv_context.get('count', 0)} CSV files")
         # ALWAYS delegate CSV tasks with full context
         result = data_analyst(f"CSV CONTEXT: {csv_context}\n\nUSER REQUEST: {user_query}")
     else:
         print("⚠️ No CSV context found - delegating anyway")
         result = data_analyst(user_query)
     
     final_answer(result)
     ```

2. **PDF/Document Questions** → use RAG tools directly:
    - Keywords: "document", "PDF", "rechercher dans", "contenu du fichier"
    - Pattern:
      ```python
      # Check for PDF context
      print("🔍 Checking for PDF context...")
      pdf_available = 'pdf_context' in globals()
      
      if pdf_available:
          pdf_context = globals()['pdf_context']
          print(f"📄 Found {pdf_context.get('count', 0)} PDF files")
          result = search_pdf_with_context(user_query, pdf_context)
      else:
          print("⚠️ No PDF context - using state search")
          result = search_pdf_from_state(user_query)
      final_answer(result)
      ```

3. **Web Research** → use web tools directly:
    - Keywords: "rechercher sur internet", "informations récentes", "actualités"
    - Use DuckDuckGoSearchTool for search, then enhanced_visit_webpage for details

MANDATORY CONTEXT CHECK PATTERN:
Before ANY action, always check context:
```python
# ALWAYS start with context check
print("🔍 CONTEXT CHECK:")
print("Available variables:", [k for k in globals().keys() if not k.startswith('_')])

csv_available = 'csv_context' in globals()
pdf_available = 'pdf_context' in globals()

print(f"📊 CSV context: {csv_available}")
print(f"📄 PDF context: {pdf_available}")

# For ANY data-related keywords → IMMEDIATE delegation
data_keywords = ['analyser', 'dataset', 'csv', 'données', 'visualisation', 'graphique', 'statistiques', 'tableau', 'colonnes', 'lignes', 'tendance', 'corrélation', 'moyenne', 'médiane', 'distribution', 'plot', 'chart']
needs_data_analysis = any(keyword.lower() in user_query.lower() for keyword in data_keywords)

if needs_data_analysis or csv_available:
    print("🎯 DELEGATING TO DATA ANALYST")
    if csv_available:
        csv_context = globals()['csv_context']
        result = data_analyst(f"CSV CONTEXT: {csv_context}\n\nUSER REQUEST: {user_query}")
    else:
        result = data_analyst(user_query)
    final_answer(result)
```

RESPONSE STYLE:
Always respond in natural, conversational French. Synthesize information from multiple sources when relevant."""

        # Create the enhanced manager agent
        manager_agent = CodeAgent(
            tools=all_tools,
            model=self.model,
            managed_agents=[data_analyst_agent],  # Only manage data analyst now
            name="enhanced_manager_agent",
            description=enhanced_description,
            stream_outputs=config.stream_outputs,
            max_steps=config.max_steps + 5,  # Slightly more steps since it does more work
            verbosity_level=config.verbosity_level,
            planning_interval=config.planning_interval,
            additional_authorized_imports=[
                "json", "re", "urllib.parse", "requests", "concurrent.futures",
                "os", "langchain_community.vectorstores"
            ]
        )
        
        print(f"✅ Enhanced manager agent created with {len(all_tools)} tools")
        return manager_agent


def create_simplified_agent_system(model: OpenAIServerModel) -> tuple[CodeAgent, CodeAgent]:
    """
    Create a simplified agent system with enhanced manager and data analyst.
    
    Args:
        model: The OpenAI model to use for all agents
        
    Returns:
        Tuple of (enhanced_manager_agent, data_analyst_agent)
    """
    factory = SimplifiedAgentFactory(model)
    
    # Create data analyst agent first
    data_analyst_agent = factory.create_data_analyst_agent()
    
    # Create enhanced manager agent with all tools
    enhanced_manager_agent = factory.create_enhanced_manager_agent(data_analyst_agent)
    
    print("🚀 Simplified agent system created successfully!")
    print(f"   - Enhanced Manager: {len(enhanced_manager_agent.tools)} tools")
    print(f"   - Data Analyst: {len(data_analyst_agent.tools)} tools")
    
    return enhanced_manager_agent, data_analyst_agent 