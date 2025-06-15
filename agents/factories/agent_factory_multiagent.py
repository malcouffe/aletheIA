"""
Multi-Agent Factory Following Smolagents Best Practices
Creates a manager agent with minimal tools that delegates to specialized agents.
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
    DATA_ANALYST_IMPORTS
)
from ..core.embedding import get_embedding_function
from ..tools import (
    data_loader,
    display_figures,
    unified_pdf_search_and_analyze
)
from ..tools.enhanced_web_tools import (
    enhanced_visit_webpage,
    bulk_visit_webpages,
    extract_financial_data
)


class MultiAgentFactory:
    """Factory following smolagents best practices: minimal manager + specialized agents."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.embedding_function = get_embedding_function()
    
    def create_data_analyst_agent(self) -> CodeAgent:
        """Create specialized data analyst agent."""
        config = AGENT_CONFIGS["data_analyst"]
        
        agent = CodeAgent(
            tools=[data_loader, display_figures],
            model=self.model,
            additional_authorized_imports=DATA_ANALYST_IMPORTS,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="data_analyst",
            description=AGENT_DESCRIPTIONS["data_analyst"],
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

        agent.prompt_templates["system_prompt"] = custom_data_analyst_prompt
        return agent

    def create_document_agent(self) -> ToolCallingAgent:
        """Create document agent with unified PDF search and analysis capabilities."""
        config = AGENT_CONFIGS["search_agent"]  # Reuse search agent config for simplicity
        
        agent = ToolCallingAgent(
            tools=[unified_pdf_search_and_analyze],
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="document_agent",
            description=AGENT_DESCRIPTIONS["rag_agent"],  # Using rag_agent description
            planning_interval=config.planning_interval
        )
        
        return agent

    def create_search_agent(self) -> ToolCallingAgent:
        """Create specialized search agent for web research."""
        config = AGENT_CONFIGS["search_agent"]
        
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
            description=AGENT_DESCRIPTIONS["search_agent"],
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


def create_multiagent_system(model: OpenAIServerModel) -> Tuple[CodeAgent, CodeAgent, ToolCallingAgent, ToolCallingAgent]:
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