"""
Multi-Agent Factory Following Smolagents Best Practices
Creates a manager agent with minimal tools that delegates to specialized agents.
"""

import streamlit as st
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, OpenAIServerModel
from typing import Optional

from ..config.agent_config import AGENT_CONFIGS, AGENT_DESCRIPTIONS
from ..tools import (
    # Unified data tools
    data_loader, display_figures,
    # RAG tools - unified tool only
    unified_pdf_search_and_analyze
)
from ..tools.enhanced_web_tools import enhanced_visit_webpage, bulk_visit_webpages, extract_financial_data
from ..core.embedding import get_embedding_function


class MultiAgentFactory:
    """Factory following smolagents best practices: minimal manager + specialized agents."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.embedding_function = get_embedding_function()
    
    def create_data_analyst_agent(self) -> CodeAgent:
        """Create specialized data analyst agent (unchanged from current implementation)."""
        config = AGENT_CONFIGS["data_analyst"]
        
        data_analyst_description = """Expert data analyst specialized in comprehensive data analysis and visualization.

CORE CAPABILITIES:
- **Data Loading & Processing**: Load CSV files, clean data, handle missing values
- **Statistical Analysis**: Descriptive statistics, correlations, hypothesis testing
- **Data Visualization**: Create plots with matplotlib, seaborn, and plotly
- **Advanced Analytics**: Clustering, classification, regression analysis
- **Data Transformation**: Grouping, pivoting, merging datasets

AVAILABLE TOOLS:
- data_loader(): Unified tool for loading CSV files and discovering available data
- display_figures(): Unified tool for displaying matplotlib and plotly visualizations

TASK HANDLING APPROACH:
1. Always start by understanding the data structure
2. Perform exploratory data analysis (EDA)
3. Apply appropriate statistical methods
4. Create clear, informative visualizations
5. Provide actionable insights and conclusions

RESPONSE FORMAT:
Always structure your response in four parts:
1. Thought: Your reasoning about what to do
2. Action: The action to take
3. Action Input: The input for the action
4. Observation: The result of the action

EXAMPLE:
Thought: I need to analyze the correlation between age and survival rate
Action: analyze_data
Action Input: {"method": "correlation", "columns": ["age", "survived"]}
Observation: The correlation analysis shows a negative correlation of -0.077 between age and survival...

CRITICAL TOOL USAGE:
- After creating ANY chart: IMMEDIATELY call display_figures() with appropriate figure_type
- Use print() statements to log important findings for debugging
- Handle errors gracefully and provide helpful troubleshooting information"""

        agent = CodeAgent(
            tools=[data_loader, display_figures],
            model=self.model,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib.pyplot", "seaborn", 
                "plotly.express", "plotly.graph_objects", "plotly.subplots", "scipy.stats",
                "sklearn.preprocessing", "sklearn.cluster", "warnings",
                "streamlit", "os", "sys", "glob",
                "datetime", "math", "statistics", "json", "csv",
                "matplotlib.patches", "matplotlib.colors", "pandas.api.types",
                "scipy.cluster", "scipy.optimize", "sklearn.metrics", "sklearn.model_selection"
            ],
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="data_analyst",
            description=data_analyst_description,
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval,
            use_structured_outputs_internally=config.use_structured_outputs_internally
        )
        
        # Set custom system prompt using smolagents best practice method
        custom_data_analyst_prompt = """You are an expert data analyst.

RESPONSE FORMAT:
Always structure your response in four parts:
1. Thought: Your reasoning about what to do
2. Action: The action to take
3. Action Input: The input for the action
4. Observation: The result of the action

{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
{%- endfor %}

{%- if managed_agents and managed_agents.values() | list %}
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
{%- endif %}

Rules: Use imports from: {{authorized_imports}}"""

        # Apply custom system prompt using smolagents method
        agent.prompt_templates["system_prompt"] = custom_data_analyst_prompt
        
        return agent

    def create_document_agent(self) -> ToolCallingAgent:
        """Create document agent with unified PDF search and analysis capabilities."""
        from ..tools.rag_tools import unified_pdf_search_and_analyze
        
        config = AGENT_CONFIGS["search_agent"]  # Reuse search agent config for simplicity
        
        document_description = """Expert PDF document analyst specialized in retrieving and analyzing content from indexed documents.

CORE MISSION: Search PDF documents and provide structured responses with citations.

TOOL AVAILABLE:
- unified_pdf_search_and_analyze(query): Search and analyze PDF content

INSTRUCTIONS:
1. Always call unified_pdf_search_and_analyze() with the user's question
2. Follow the structured response format
3. The tool already handles citations [1], [2], etc. and sources

RESPONSE FORMAT:
Always structure your response in four parts:
1. Thought: Your reasoning about what to do
2. Action: The action to take
3. Action Input: The input for the action
4. Observation: The result of the action

EXAMPLE:
Thought: I need to search for information about internal controls
Action: unified_pdf_search_and_analyze
Action Input: {"query": "internal controls"}
Observation: [Tool output with citations and sources]"""

        agent = ToolCallingAgent(
            tools=[unified_pdf_search_and_analyze],
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="document_agent",
            description=document_description,
            planning_interval=config.planning_interval
        )
        
        return agent

    def create_search_agent(self) -> ToolCallingAgent:
        """Create specialized search agent for web research."""
        config = AGENT_CONFIGS["search_agent"]
        
        search_description = """Expert web researcher specializing in comprehensive information gathering.

CORE CAPABILITIES:
- Web search using DuckDuckGo
- Enhanced webpage analysis with content extraction
- Bulk webpage processing for comprehensive research
- Financial data extraction from web sources

AVAILABLE TOOLS:
- DuckDuckGoSearchTool(): Web search functionality
- enhanced_visit_webpage(): Deep webpage content analysis
- bulk_visit_webpages(): Process multiple pages efficiently
- extract_financial_data(): Extract financial information

RESEARCH APPROACH:
1. Use targeted search queries for best results
2. Visit and analyze relevant webpages
3. Extract and synthesize information from multiple sources
4. Provide comprehensive findings with sources

TOOL USAGE BEST PRACTICES:
- Start with broad searches, then narrow down with specific queries
- Visit 2-3 most relevant pages for comprehensive coverage
- Use bulk_visit_webpages() for multiple related URLs
- Always cite sources in your response

RESPONSE FORMAT:
Always structure your response in four parts:
1. Thought: Your reasoning about what to do
2. Action: The action to take
3. Action Input: The input for the action
4. Observation: The result of the action"""

        # Use ToolCallingAgent for web search as it's more suitable for single-timeline tasks
        web_tools = []
        try:
            # Configuration simple sans paramètres pour éviter les erreurs
            search_tool = DuckDuckGoSearchTool()
            web_tools.extend([
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages,
                extract_financial_data
            ])
        except ImportError as e:
            print(f"⚠️ Some web tools unavailable: {e}")
            # Fallback sans DuckDuckGo si problème
            web_tools.extend([
                enhanced_visit_webpage,
                bulk_visit_webpages,
                extract_financial_data
            ])

        agent = ToolCallingAgent(
            tools=web_tools,
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="search_agent",
            description=search_description,
            planning_interval=config.planning_interval
        )
        
        return agent

    def create_minimal_manager_agent(self, managed_agents: list) -> CodeAgent:
        """Create minimal manager agent following smolagents best practices."""
        config = AGENT_CONFIGS["manager_agent"]
        
        # PURE DELEGATION - no execution tools as per smolagents best practices
        coordination_tools = []  # Manager = pure delegation, no execution
        
        agent = CodeAgent(
            tools=coordination_tools,
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="manager_agent",
            description=AGENT_DESCRIPTIONS["manager_agent"],
            managed_agents=managed_agents,
            planning_interval=config.planning_interval,
            use_structured_outputs_internally=config.use_structured_outputs_internally
        )
        
        return agent


def create_multiagent_system(model: OpenAIServerModel) -> tuple[CodeAgent, CodeAgent, ToolCallingAgent, ToolCallingAgent]:
    """
    Create a multi-agent system following smolagents best practices.
    
    Args:
        model: OpenAI model to use for all agents
        
    Returns:
        Tuple of (manager_agent, data_analyst_agent, document_agent, search_agent)
    """
    factory = MultiAgentFactory(model)
    
    # Create specialized agents first
    data_analyst = factory.create_data_analyst_agent()
    document_agent = factory.create_document_agent()
    search_agent = factory.create_search_agent()
    
    # Create minimal manager agent that delegates to specialized agents
    manager = factory.create_minimal_manager_agent([
        data_analyst,
        document_agent,
        search_agent
    ])
    
    return manager, data_analyst, document_agent, search_agent 