"""
Multi-Agent Factory Following Smolagents Best Practices
Creates a manager agent with minimal tools that delegates to specialized agents.
"""

import streamlit as st
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, OpenAIServerModel
from typing import Optional

from ..config.agent_config import AGENT_CONFIGS, AGENT_DESCRIPTIONS
from ..tools import (
    # Data visualization tools
    display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files,
    # RAG tools (Streamlit optimized) - ESSENTIAL ONLY
    search_pdf_interactive, smart_pdf_search_for_streamlit, display_source_passages,
    validate_source_display_before_final_answer
)
from ..tools.context_access_tools import check_context_availability, demonstrate_context_access
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
- load_csv_data(): Load and examine CSV files with pandas
- discover_data_files(): Find available data files in the workspace
- display_matplotlib_figures(): Show matplotlib plots to user
- display_plotly_figures(): Show interactive plotly visualizations

TASK HANDLING APPROACH:
1. Always start by understanding the data structure
2. Perform exploratory data analysis (EDA)
3. Apply appropriate statistical methods
4. Create clear, informative visualizations
5. Provide actionable insights and conclusions

RESPONSE STYLE (CRITICAL):
- Always respond in natural, conversational French
- Explain findings as if discussing with a colleague
- Use flowing paragraphs, not structured data
- Never return dictionaries or technical formats in final_answer()
- Tell a story about what the data reveals
- Make insights accessible and engaging

EXAMPLE GOOD RESPONSE:
"J'ai analysé le dataset Titanic et voici ce que j'ai découvert. Sur les 891 passagers, environ 38% ont survécu au naufrage. L'âge moyen était de 30 ans environ. Ce qui est frappant, c'est que la classe sociale jouait un rôle important..."

EXAMPLE BAD RESPONSE:
{"Total Passengers": 891, "Percentage": 38.38, ...}

CRITICAL TOOL USAGE:
- After creating ANY chart: IMMEDIATELY call display_matplotlib_figures() or display_plotly_figures()
- Use print() statements to log important findings for debugging
- Handle errors gracefully and provide helpful troubleshooting information"""

        agent = CodeAgent(
            tools=[display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files],
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
            planning_interval=config.planning_interval
        )
        
        # Set custom system prompt using smolagents best practice method
        custom_data_analyst_prompt = """You are an expert data analyst.

CRITICAL: Always respond in natural, conversational French. Never return structured data or dictionaries in final_answer().

Example good response: "J'ai analysé le dataset et découvert que 38% des passagers ont survécu. L'âge moyen était de 30 ans..."
Example bad response: {"Total": 891, "Percentage": 38.38}

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

Rules: Always provide 'Thought:' and 'Code:' sequences. Use imports from: {{authorized_imports}}"""

        # Apply custom system prompt using smolagents method
        agent.prompt_templates["system_prompt"] = custom_data_analyst_prompt
        
        print(f"✅ Data Analyst Agent created")
        return agent

    def create_rag_agent(self) -> CodeAgent:
        """Create simplified RAG agent following smolagents best practices."""
        config = AGENT_CONFIGS["rag_agent"]
        
        rag_description = """PDF document search tool wrapper. ONLY calls search_pdf_interactive() and returns its output.

CRITICAL: I have NO knowledge about documents. I MUST ALWAYS call search_pdf_interactive() for ANY question about PDFs.

WORKFLOW:
1. User asks PDF question → call search_pdf_interactive(exact_query)
2. Return tool output unchanged
3. Never add my own content or analysis

FORBIDDEN: I cannot answer PDF questions without calling the tool first."""

        # No need to import unified tool - using only search_pdf_interactive

        agent = CodeAgent(
            tools=[
                search_pdf_interactive,
            ],
            model=self.model,
            additional_authorized_imports=[
                "json", "re", "os", "langchain_community.vectorstores", "streamlit"
            ],
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="rag_agent",
            description=rag_description,
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval
        )
        
        # Set custom system prompt for NotebookLM behavior
        rag_prompt = """You are a tool wrapper agent. You have NO knowledge about documents.

AVAILABLE TOOLS:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
{%- endfor %}

CRITICAL RULES:
🚨 I CANNOT answer questions about PDFs without calling search_pdf_interactive() FIRST
🚨 I have NO document knowledge stored in my memory
🚨 I MUST call the tool for EVERY PDF question

MANDATORY WORKFLOW:
1. User asks PDF question
2. IMMEDIATELY call: search_pdf_interactive("exact user query")
3. Return the tool result UNCHANGED
4. Stop - do not add anything

EXAMPLE:
User: "What does the document say about governance?"
```python
result = search_pdf_interactive("What does the document say about governance?")
final_answer(result)
```

NEVER answer from memory. ALWAYS call the tool first."""

        # Apply custom system prompt
        agent.prompt_templates["system_prompt"] = rag_prompt
        
        print(f"✅ RAG Agent created with NotebookLM workflow (1 tool: search_pdf_interactive only)")
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

RESPONSE STYLE:
Always respond in natural, conversational French. Explain findings as if discussing with a colleague.
Never return raw search results - synthesize into flowing, natural explanations.

SYNTHESIS GUIDELINES:
- Combine information from multiple sources
- Highlight important trends or patterns
- Provide context and background when relevant
- Indicate confidence levels and source reliability"""

        # Use ToolCallingAgent for web search as it's more suitable for single-timeline tasks
        web_tools = []
        try:
            search_tool = DuckDuckGoSearchTool()
            web_tools.extend([
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages,
                extract_financial_data
            ])
        except ImportError as e:
            print(f"⚠️ Some web tools unavailable: {e}")

        agent = ToolCallingAgent(
            tools=web_tools,
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="search_agent",
            description=search_description,
            planning_interval=config.planning_interval
        )
        
        print(f"✅ Search Agent created")
        return agent

    def create_minimal_manager_agent(self, managed_agents: list) -> CodeAgent:
        """Create minimal manager agent following smolagents best practices."""
        config = AGENT_CONFIGS["manager_agent"]
        
        # MINIMAL tools - only essential coordination
        coordination_tools = []  # No tools needed - just pure delegation
        
        manager_description = """Minimal coordination manager following smolagents best practices - DELEGATE IMMEDIATELY.

CORE PRINCIPLE: Act as a smart switchboard operator - identify the right specialist and delegate instantly.

ROUTING DECISION TREE (apply in strict order):

1. DATA ANALYSIS/STATISTICS → data_analyst:
   - Trigger words: "analyser", "analyse", "données", "dataset", "CSV", "Excel", "Titanic"
   - Trigger words: "statistiques", "corrélation", "graphique", "visualisation", "chart"
   - Trigger words: "moyenne", "médiane", "distribution", "tendances", "insights"

2. PDF DOCUMENT SEARCH → rag_agent:
   - Trigger words: "document", "PDF", "fichier", "rechercher dans", "contenu"
   - Trigger words: "rapport", "article", "citation", "référence", "résumé"
   - Trigger words: "trouver", "localiser", "extraire", "information spécifique"

3. WEB RESEARCH/CURRENT INFO → search_agent:
   - Trigger words: "rechercher", "internet", "web", "actualités", "informations récentes"
   - Trigger words: "prix", "cours", "actions", "nouvelles", "vérifier", "confirmer"

CRITICAL RULES:
✅ ALWAYS delegate immediately - never analyze or solve yourself
✅ Pass the COMPLETE user query to the specialist
✅ Trust specialists completely
✅ Keep steps minimal (max 2) - delegate in first step

❌ NEVER attempt specialized work yourself

DELEGATION: Use the exact agent name (data_analyst, rag_agent, search_agent) to delegate tasks."""

        manager_agent = CodeAgent(
            tools=coordination_tools,  # No tools - just delegation
            model=self.model,
            managed_agents=managed_agents,  # Proper smolagents delegation
            name="coordination_manager",
            description=manager_description,
            stream_outputs=config.stream_outputs,
            max_steps=config.max_steps,  # Keep low due to delegation
            verbosity_level=config.verbosity_level,
            planning_interval=config.planning_interval,
            additional_authorized_imports=[]  # No imports needed for delegation
        )
        
        # Set custom system prompt using smolagents best practice method
        minimal_manager_prompt = """You are a minimal coordination manager. Your ONLY job is to delegate to specialist agents immediately.

AVAILABLE SPECIALIST AGENTS:
{%- if managed_agents and managed_agents.values() | list %}
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
{%- endif %}

ROUTING DECISION TREE (apply these rules in strict order):

1. DATA/ANALYSIS/STATISTICS → delegate to data_analyst:
   Keywords: "analyser", "analyse", "données", "dataset", "CSV", "Excel", "Titanic", "statistiques", "corrélation", "graphique", "visualisation", "chart", "moyenne", "médiane", "distribution", "tendances", "insights"

2. PDF/DOCUMENTS → delegate to rag_agent:
   Keywords: "document", "PDF", "fichier", "rechercher dans", "contenu", "rapport", "article", "citation", "référence", "résumé", "trouver", "localiser", "extraire"

3. WEB/RESEARCH/CURRENT INFO → delegate to search_agent:
   Keywords: "rechercher", "internet", "web", "actualités", "informations récentes", "prix", "cours", "actions", "nouvelles", "vérifier", "confirmer", "comparaison", "concurrence"

CRITICAL INSTRUCTIONS:
1. SCAN the user request for trigger keywords
2. IMMEDIATELY identify which specialist agent to use
3. DELEGATE using the exact agent name: data_analyst(task="FULL user request")
4. RETURN the agent's response directly
5. NEVER attempt to solve tasks yourself
6. NEVER analyze or process before delegation
7. Keep your response to maximum 2 steps: decide + delegate

DELEGATION EXAMPLES:
- User: "Analyse le dataset Titanic" → data_analyst(task="Analyse le dataset Titanic")
- User: "Recherche dans les PDFs" → rag_agent(task="Recherche dans les PDFs") 
- User: "Trouve des infos sur internet" → search_agent(task="Trouve des infos sur internet")

{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
{%- endfor %}

Remember: You are a conductor pointing to musicians, not playing the instruments yourself. Always provide 'Thought:' and 'Code:' sequences. Delegate immediately without hesitation."""

        # Apply custom system prompt using smolagents method
        manager_agent.prompt_templates["system_prompt"] = minimal_manager_prompt
        
        print(f"✅ Minimal Manager created with {len(coordination_tools)} tools and {len(managed_agents)} managed agents")
        return manager_agent


def create_multiagent_system(model: OpenAIServerModel) -> tuple[CodeAgent, CodeAgent, CodeAgent, ToolCallingAgent]:
    """
    Create a multi-agent system following smolagents best practices.
    
    SMOLAGENTS COMPLIANCE:
    - Manager has minimal tools (only 2 coordination tools)
    - Each agent is specialized for one domain
    - Clear delegation hierarchy with managed_agents parameter
    - Reduced complexity and LLM calls
    
    Args:
        model: The OpenAI model to use for all agents
        
    Returns:
        Tuple of (manager, data_analyst, rag_agent, search_agent)
    """
    factory = MultiAgentFactory(model)
    
    # Create specialized agents first
    data_analyst = factory.create_data_analyst_agent()
    rag_agent = factory.create_rag_agent()
    search_agent = factory.create_search_agent()
    
    # Create minimal manager with all specialized agents
    manager = factory.create_minimal_manager_agent([
        data_analyst, rag_agent, search_agent
    ])
    
    print("🚀 Multi-Agent System created following smolagents best practices!")
    print(f"   - Manager: {len(manager.tools)} tools + {len(manager.managed_agents)} agents")
    print(f"   - Data Analyst: {len(data_analyst.tools)} specialized tools")
    print(f"   - RAG Agent: {len(rag_agent.tools)} document tools")
    print(f"   - Search Agent: {len(search_agent.tools)} web tools")
    print("   - Follows smolagents principle: 'simplest systems are best'")
    
    return manager, data_analyst, rag_agent, search_agent 