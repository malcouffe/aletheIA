"""
Multi-Agent Factory pour Système Simplifié
Crée uniquement les agents spécialisés (plus de manager agent).
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import streamlit as st
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    OpenAIServerModel
)

# Local imports
from ..config.agent_config import (
    AGENT_CONFIGS,
    AGENT_DESCRIPTIONS,
    DATA_ANALYST_IMPORTS,
    SYSTEM_COMMUNICATION_PROMPT,
    USER_QUERY_PREPROMPTS,
    PREPROMPT_CONFIG
)
from ..core.embedding import get_embedding_function
from ..tools.unified_data_tools import load_and_explore_csv, display_figures
from ..tools import (
    rag_search_simple,
    data_loader,
    get_dataframe
)
from ..tools.enhanced_web_tools import (
    enhanced_visit_webpage,
    bulk_visit_webpages,
    extract_financial_data
)


class SimplifiedAgentFactory:
    """Factory pour le système simplifié : seulement les agents spécialisés."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.embedding_function = get_embedding_function()
    
    def create_data_analyst_agent(self) -> CodeAgent:
        """Create specialized data analyst agent."""
        config = AGENT_CONFIGS["data_analyst"]
        
        # Intégrer le prompt système pour forcer le français naturel
        enhanced_description = SYSTEM_COMMUNICATION_PROMPT + "\n\n" + AGENT_DESCRIPTIONS["data_analyst"]
        
        agent = CodeAgent(
            tools=[load_and_explore_csv, display_figures, data_loader, get_dataframe],
            model=self.model,
            additional_authorized_imports=DATA_ANALYST_IMPORTS,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="data_analyst",
            description=enhanced_description,
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval,
            use_structured_outputs_internally=config.use_structured_outputs_internally
        )
        
        return agent

    def create_document_agent(self) -> ToolCallingAgent:
        """Create document agent with unified PDF search and analysis capabilities."""
        config = AGENT_CONFIGS["rag_agent"]
        
        # Intégrer le prompt système pour forcer le français naturel
        enhanced_description = SYSTEM_COMMUNICATION_PROMPT + "\n\n" + AGENT_DESCRIPTIONS["rag_agent"]
        
        agent = ToolCallingAgent(
            tools=[rag_search_simple],
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="document_agent",
            description=enhanced_description,
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval
        )

        return agent

    def create_search_agent(self) -> CodeAgent:
        """Create specialized search agent for web research."""
        config = AGENT_CONFIGS["search_agent"]
        
        # Intégrer le prompt système pour forcer le français naturel
        enhanced_description = SYSTEM_COMMUNICATION_PROMPT + "\n\n" + AGENT_DESCRIPTIONS["search_agent"]
        
        web_tools = []
        try:
            search_tool = DuckDuckGoSearchTool()
            web_tools.extend([
                search_tool,
                enhanced_visit_webpage,
                bulk_visit_webpages,
                extract_financial_data
            ])
        except ImportError:
            web_tools.extend([
                enhanced_visit_webpage,
                bulk_visit_webpages,
                extract_financial_data
            ])

        agent = CodeAgent(
            tools=web_tools,
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="search_agent",
            description=enhanced_description,
            stream_outputs=config.stream_outputs,
            planning_interval=config.planning_interval,
            use_structured_outputs_internally=config.use_structured_outputs_internally
        )

        return agent


def create_multiagent_system(model: OpenAIServerModel) -> Tuple[None, CodeAgent, ToolCallingAgent, CodeAgent]:
    """
    Create a simplified multi-agent system (no manager agent).
    
    Args:
        model: OpenAI model to use for all agents
        
    Returns:
        Tuple of (None, data_analyst_agent, document_agent, search_agent)
        Note: Premier élément None car plus de manager agent
    """
    factory = SimplifiedAgentFactory(model)
    
    # Create specialized agents only
    data_analyst = factory.create_data_analyst_agent()
    document_agent = factory.create_document_agent()
    search_agent = factory.create_search_agent()
    
    # Return None for manager agent (backward compatibility)
    return None, data_analyst, document_agent, search_agent


# Backward compatibility - keep old class name as alias
MultiAgentFactory = SimplifiedAgentFactory 