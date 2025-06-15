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
        
        # Prompt template en Français pour l'agent data_analyst
#         custom_data_analyst_prompt = """
# Vous êtes un expert en analyse de données.

# FORMAT DE RÉPONSE :
# Structurez toujours votre réponse en quatre parties :
# 1. Thought : votre raisonnement sur l'étape suivante
# 2. Action : l'action à exécuter
# 3. Action Input : les paramètres JSON pour l'action
# 4. Observation : le résultat obtenu

# {%- for tool in tools.values() %}
# - {{ tool.name }} : {{ tool.description }}
#   Prend en entrée : {{ tool.inputs }}
#   Retourne : {{ tool.output_type }}
# {%- endfor %}

# {%- if managed_agents and managed_agents.values() | list %}
# Agents gérés :
# {%- for agent in managed_agents.values() %}
# - {{ agent.name }} : {{ agent.description }}
# {%- endfor %}
# {%- endif %}

# Règle : utilisez uniquement les imports autorisés : {{ authorized_imports }}
# """
#         agent.prompt_templates["system_prompt"] = custom_data_analyst_prompt
        return agent

    def create_document_agent(self) -> ToolCallingAgent:
        """Create document agent with unified PDF search and analysis capabilities."""
        config = AGENT_CONFIGS["rag_agent"]
        
        agent = ToolCallingAgent(
            tools=[unified_pdf_search_and_analyze],
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="document_agent",
            description=AGENT_DESCRIPTIONS["rag_agent"],
            planning_interval=config.planning_interval
        )

        # Prompt template en Français pour l'agent document_agent
#         custom_document_prompt = """
# Vous êtes un spécialiste de l'analyse de documents PDF.

# FORMAT DE RÉPONSE :
# 1. Thought : votre réflexion
# 2. Action : unified_pdf_search_and_analyze
# 3. Action Input : {"query": "<votre requête>"}
# 4. Observation : extraits issus du PDF avec citations

# OUTIL :
# - unified_pdf_search_and_analyze(query) : recherche et analyse dans les PDF indexés

# Règle : citez toujours chaque passage au format [1], [2], ...
# """
#         agent.prompt_templates["system_prompt"] = custom_document_prompt
        return agent

    def create_search_agent(self) -> ToolCallingAgent:
        """Create specialized search agent for web research."""
        config = AGENT_CONFIGS["search_agent"]
        
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

        agent = ToolCallingAgent(
            tools=web_tools,
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="search_agent",
            description=AGENT_DESCRIPTIONS["search_agent"],
            planning_interval=config.planning_interval
        )

        # Prompt template en Français pour l'agent search_agent
#         custom_search_prompt = """
# Vous êtes un expert en recherche web et synthèse d'information.

# FORMAT DE RÉPONSE :
# 1. Thought : votre raisonnement pour la recherche
# 2. Action : l'outil à utiliser
# 3. Action Input : paramètres JSON de l'outil
# 4. Observation : résultats et sources

# OUTILS DISPONIBLES :
# {%- for tool in tools.values() %}
# - {{ tool.name }} : {{ tool.description }}
# {%- endfor %}

# Règle : citez toujours chaque information avec une source claire.
# """
#         agent.prompt_templates["system_prompt"] = custom_search_prompt
        return agent

    def create_minimal_manager_agent(self, managed_agents: list) -> CodeAgent:
        """Create minimal manager agent following smolagents best practices."""
        config = AGENT_CONFIGS["manager_agent"]
        
        agent = CodeAgent(
            tools=[],  # aucun outil, pure délégation
            model=self.model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            name="manager_agent",
            description=AGENT_DESCRIPTIONS["manager_agent"],
            managed_agents=managed_agents,
            planning_interval=config.planning_interval,
            use_structured_outputs_internally=config.use_structured_outputs_internally
        )

        # Prompt template en Français pour l'agent manager_agent
#         custom_manager_prompt = """
# Vous êtes un agent manager dont le SEUL rôle est de déléguer les tâches aux agents spécialisés.

# AGENTS DISPONIBLES :
# {%- for agent in managed_agents.values() %}
# - {{ agent.name }} : {{ agent.description }}
# {%- endfor %}

# FORMAT DE RÉPONSE OBLIGATOIRE :
# 1. Thought : votre raisonnement sur le choix de l'agent spécialisé
# 2. Action : nom de l'agent spécialisé à appeler
# 3. Action Input : la requête utilisateur complète à transmettre
# 4. Observation : résultat retourné par l'agent spécialisé

# EXEMPLES DE DÉLÉGATION :

# 1. Analyse de données CSV :
# Thought: La requête concerne l'analyse d'un dataset CSV
# Action: data_analyst
# Action Input: "Analyse le dataset bank_transaction"
# Observation: [résultat de data_analyst]

# 2. Recherche dans des documents PDF :
# Thought: La requête concerne la recherche dans des documents PDF
# Action: document_agent
# Action Input: "Trouve les informations sur les contrôles internes dans les rapports"
# Observation: [résultat de document_agent]

# 3. Recherche web :
# Thought: La requête nécessite une recherche d'informations sur le web
# Action: search_agent
# Action Input: "Trouve les dernières informations sur les régulations bancaires"
# Observation: [résultat de search_agent]

# 4. Analyse de données avec visualisation :
# Thought: La requête demande une analyse avec des graphiques
# Action: data_analyst
# Action Input: "Crée des visualisations pour le dataset bank_transaction"
# Observation: [résultat de data_analyst]

# RÈGLES CRITIQUES :
# - DÉLÉGUER IMMÉDIATEMENT - ne jamais exécuter de code
# - Ne jamais modifier la requête utilisateur
# - Ne jamais traiter la tâche directement
# - Toujours utiliser le format de réponse exact ci-dessus
# - Toujours choisir l'agent le plus approprié pour la tâche
# """
#         agent.prompt_templates["system_prompt"] = custom_manager_prompt
        return agent

def create_multiagent_system(model: OpenAIServerModel) -> Tuple[ToolCallingAgent, CodeAgent, ToolCallingAgent, ToolCallingAgent]:
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