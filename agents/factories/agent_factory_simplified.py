"""
Simplified Agent Factory for Streamlined Architecture
Creates only Manager Agent (with all tools) and Data Analyst Agent.
"""

import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, tool
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
        
        # Detailed description following smolagents best practices
        data_analyst_description = """Expert data analyst specialized in comprehensive data analysis and visualization.

CORE CAPABILITIES:
- **Data Loading & Processing**: Load CSV files, clean data, handle missing values
- **Statistical Analysis**: Descriptive statistics, correlations, hypothesis testing
- **Data Visualization**: Create plots with matplotlib, seaborn, and plotly
- **Advanced Analytics**: Clustering, classification, regression analysis
- **Data Transformation**: Grouping, pivoting, merging datasets

AVAILABLE TOOLS:
- load_csv_data(): Load and examine CSV files with pandas
- discover_data_files(): Find available data files in the workspace
- display_matplotlib_figures(): Show matplotlib plots to user
- display_plotly_figures(): Show interactive plotly visualizations

AUTHORIZED LIBRARIES:
pandas, numpy, matplotlib, seaborn, plotly, scipy, sklearn, datetime, statistics

TASK HANDLING APPROACH:
1. Always start by understanding the data structure
2. Perform exploratory data analysis (EDA)
3. Apply appropriate statistical methods
4. Create clear, informative visualizations
5. Provide actionable insights and conclusions

IMPORTANT EXECUTION NOTES:
- Use print() statements to log progress and findings
- Always explain statistical results in plain language
- Create multiple visualization types when relevant
- Handle data quality issues proactively
- Respond in natural, conversational French

Example task handling:
- "Analyze sales data" → Load data, examine columns, create summary statistics, plot trends
- "Compare categories" → Group by category, calculate metrics, create comparison charts
- "Find patterns" → Apply statistical tests, create correlation matrices, identify outliers"""

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
            description=data_analyst_description,  # Much more detailed description
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval
        )
        
        print(f"✅ Data analyst agent created successfully with detailed capabilities")
        return agent

    def create_enhanced_manager_agent(self, data_analyst_agent: CodeAgent) -> CodeAgent:
        """Create an enhanced manager agent with proper smolagent delegation."""
        config = AGENT_CONFIGS["manager_agent"]
        
        # Simplified tools: ONLY essential coordination tools
        coordination_tools = []
        
        # Add only web search tools for coordination (not the agent's direct use)
        try:
            search_tool = DuckDuckGoSearchTool()
            coordination_tools.extend([
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages
            ])
            print("✅ Web coordination tools added to manager")
        except ImportError as ie:
            print(f"⚠️ Web tools unavailable: {ie}")
        
        # Add RAG tools for direct document access (manager's responsibility)
        try:
            coordination_tools.extend([
                search_pdf_documents,
                search_pdf_with_context,
                search_pdf_from_state,
                get_citation_help
            ])
            print("✅ RAG tools added to manager")
        except Exception as e:
            print(f"⚠️ RAG tools partially unavailable: {e}")
        
        # Enhanced manager description following smolagent best practices
        manager_description = """Expert coordination agent with clear delegation rules.

DELEGATION STRATEGY :
1. **ALWAYS delegate data analysis tasks** to data_analyst
   - Keywords: CSV, data, analysis, visualization, pandas, plotting, charts, graphs, statistics
   - Keywords: correlation, regression, clustering, EDA, exploratory, trends, patterns
   - Keywords: matplotlib, seaborn, plotly, dataframe, dataset, metrics
   - Usage: data_analyst(task="Detailed description including: data source, analysis goals, specific visualizations needed, and expected insights")
   
   DELEGATION EXAMPLES:
   ✅ "data_analyst(task='Load the sales.csv file, perform exploratory data analysis, create trend visualizations by month, and identify top-performing categories')"
   ✅ "data_analyst(task='Analyze customer data: calculate descriptive statistics, create correlation matrix, identify outliers, and visualize distributions')"
   ❌ Never attempt: "Let me load this CSV file myself" - ALWAYS delegate

2. **Handle document queries directly** using RAG tools
   - Use search_pdf_with_context() for document searches
   - Use get_citation_help() for references

3. **Handle web research directly** using web tools  
   - Use DuckDuckGoSearchTool() for searches
   - Use enhanced_visit_webpage() for page content

CRITICAL DELEGATION RULES:
- ANY data/CSV/statistical task → MUST delegate to data_analyst with very detailed instructions
- Be extremely specific in delegation: mention data source, analysis type, visualization needs
- The data_analyst has comprehensive capabilities: pandas, numpy, matplotlib, seaborn, plotly, scipy, sklearn
- Never attempt data analysis yourself - data_analyst is the expert
- Provide context and expected outcomes when delegating
- Respond in natural, conversational French

Remember: Proper delegation with detailed instructions reduces errors and improves efficiency per smolagents principles."""

        # Create the manager agent with proper smolagent delegation
        manager_agent = CodeAgent(
            tools=coordination_tools,  # Simplified tool set
            model=self.model,
            managed_agents=[data_analyst_agent],  # Correct smolagent delegation
            name="coordination_manager", 
            description=manager_description,
            stream_outputs=config.stream_outputs,
            max_steps=config.max_steps,  # Reduced steps due to delegation
            verbosity_level=config.verbosity_level,
            planning_interval=config.planning_interval,
            additional_authorized_imports=[
                "json", "re", "urllib.parse", "requests",
                "os", "langchain_community.vectorstores"
            ]
        )
        
        print(f"✅ Coordination manager created with {len(coordination_tools)} tools and 1 managed agent")
        return manager_agent


def create_simplified_agent_system(model: OpenAIServerModel) -> tuple[CodeAgent, CodeAgent]:
    """
    Create a simplified agent system following smolagents best practices.
    
    SMOLAGENTS COMPLIANCE:
    - Uses managed_agents parameter for proper delegation
    - Manager has minimal tools to reduce LLM calls 
    - Clear delegation rules in system prompt
    - Specialized agents for specific tasks
    
    Args:
        model: The OpenAI model to use for all agents
        
    Returns:
        Tuple of (coordination_manager, data_analyst_agent)
    """
    factory = SimplifiedAgentFactory(model)
    
    # Create specialized data analyst agent first
    data_analyst_agent = factory.create_data_analyst_agent()
    
    # Create coordination manager with proper smolagent delegation
    coordination_manager = factory.create_enhanced_manager_agent(data_analyst_agent)
    
    print("🚀 Smolagents-compliant system created successfully!")
    print(f"   - Coordination Manager: {len(coordination_manager.tools)} tools + 1 managed agent")
    print(f"   - Data Analyst: {len(data_analyst_agent.tools)} specialized tools")
    print("   - Follows smolagents best practices for delegation and simplicity")
    
    return coordination_manager, data_analyst_agent 