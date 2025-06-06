"""
Agent Factory for Creating Specialized Agents
Centralized agent creation with consistent configuration.
"""

import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
from typing import Optional

from ..config.agent_config import AGENT_CONFIGS, AGENT_DESCRIPTIONS
from ..tools import (
    display_matplotlib_figures, display_plotly_figures, load_csv_data, discover_data_files,
    search_pdf_documents, search_pdf_with_context, search_pdf_from_state, extract_pdf_context_and_delegate, get_citation_help,
    enhanced_visit_webpage, bulk_visit_webpages, extract_financial_data
)
from ..core.embedding import get_embedding_function


class AgentFactory:
    """Factory for creating specialized agents with consistent configuration."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.embedding_function = get_embedding_function()
    
    def create_search_agent(self) -> CodeAgent:
        """Create a web search agent with enhanced tools."""
        config = AGENT_CONFIGS["search_agent"]
        
        try:
            search_tool = DuckDuckGoSearchTool()
            
            enhanced_tools = [
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages, 
                extract_financial_data
            ]
            
            agent = CodeAgent(
                tools=enhanced_tools,
                model=self.model,
                name="search_agent",
                description=AGENT_DESCRIPTIONS["search_agent"],
                max_steps=config.max_steps,
                additional_authorized_imports=[
                    "json", "re", "urllib.parse", "requests", "concurrent.futures"
                ],
                stream_outputs=config.stream_outputs,
                planning_interval=config.planning_interval,
                verbosity_level=config.verbosity_level
            )
            
            print("✅ Enhanced search agent initialized successfully!")
            return agent
            
        except ImportError as ie:
            print(f"❌ Import error creating search agent: {ie}")
            print("Missing dependencies. Try: pip install smolagents[search] beautifulsoup4 requests")
            return self._create_fallback_agent("search_agent", "Basic agent (enhanced search tools unavailable due to missing dependencies).")
        except Exception as e:
            print(f"❌ Failed to initialize search agent: {e}")
            return self._create_fallback_agent("search_agent", "Basic agent with no tools due to initialization failure.")

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
        
        # 🔍 DEBUG: Vérifier la configuration de l'agent
        print(f"🔍 DATA ANALYST DEBUG INFO:")
        print(f"   - Agent name: {agent.name}")
        print(f"   - Max steps: {agent.max_steps}")
        print(f"   - Model: {type(agent.model).__name__}")
        print(f"   - Tools count: {len(agent.tools)}")
        print(f"   - Tools names: {list(agent.tools.keys())}")
        
        # Vérifier les imports autorisés
        if hasattr(agent, 'additional_authorized_imports'):
            print(f"   - Authorized imports: {agent.additional_authorized_imports}")
        else:
            print(f"   - ❌ NO authorized imports attribute found!")
        
        # Vérifier l'environnement d'exécution
        if hasattr(agent, 'python_executor'):
            print(f"   - Python executor type: {type(agent.python_executor).__name__}")
            if hasattr(agent.python_executor, 'authorized_imports'):
                print(f"   - Executor authorized imports: {agent.python_executor.authorized_imports}")
        
        print(f"✅ Data analyst agent created with debug info above")
        return agent

    def create_rag_agent(self) -> Optional[CodeAgent]:
        """Create a RAG agent for document retrieval and analysis."""
        config = AGENT_CONFIGS["rag_agent"]
        
        try:
            rag_agent = CodeAgent(
                tools=[search_pdf_from_state, search_pdf_with_context, get_citation_help],
                model=self.model,
                name="rag_agent",
                max_steps=config.max_steps,
                verbosity_level=config.verbosity_level,
                description=AGENT_DESCRIPTIONS["rag_agent"],
                planning_interval=config.planning_interval,
                stream_outputs=config.stream_outputs,
                additional_authorized_imports=["os", "langchain_community.vectorstores"]
            )
            print("✅ RAG Agent with state-aware PDF search tool initialized successfully!")
            return rag_agent

        except Exception as rag_init_error:
            st.error(f"Erreur lors de l'initialisation du RAG Agent : {rag_init_error}")
            print(f"RAG Agent initialization failed: {rag_init_error}")
            return None

    def create_manager_agent(
        self, 
        search_agent: CodeAgent, 
        data_analyst_agent: CodeAgent, 
        rag_agent: Optional[CodeAgent]
    ) -> CodeAgent:
        """Create a manager agent that coordinates other agents."""
        config = AGENT_CONFIGS["manager_agent"]
        
        # Prepare managed agents list (filter out None values)
        managed_agents = [agent for agent in [search_agent, data_analyst_agent, rag_agent] if agent is not None]
        
        manager_agent = CodeAgent(
            tools=[],  # Manager focuses on delegation, not direct tool use
            model=self.model,
            managed_agents=managed_agents,
            name="manager_agent",
            description=AGENT_DESCRIPTIONS["manager_agent"],
            stream_outputs=config.stream_outputs,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            planning_interval=config.planning_interval
        )
        
        return manager_agent
    
    def _create_fallback_agent(self, name: str, description: str) -> CodeAgent:
        """Create a fallback agent with minimal functionality."""
        return CodeAgent(
            tools=[],
            model=self.model,
            name=name,
            description=description,
            max_steps=3
        )


def create_agent_system(model: OpenAIServerModel) -> tuple[CodeAgent, CodeAgent, CodeAgent, Optional[CodeAgent]]:
    """
    Create a complete agent system with all specialized agents.
    
    Args:
        model: The OpenAI model to use for all agents
        
    Returns:
        Tuple of (manager_agent, search_agent, data_analyst_agent, rag_agent)
    """
    factory = AgentFactory(model)
    
    # Create specialized agents
    search_agent = factory.create_search_agent()
    data_analyst_agent = factory.create_data_analyst_agent()
    rag_agent = factory.create_rag_agent()
    
    # Create manager agent that coordinates the others
    manager_agent = factory.create_manager_agent(search_agent, data_analyst_agent, rag_agent)
    
    return manager_agent, search_agent, data_analyst_agent, rag_agent 