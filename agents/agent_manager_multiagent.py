"""
Interface de Gestion Multi-Agents
Implémente les meilleures pratiques smolagents avec des agents spécialisés et un gestionnaire minimal.
"""

from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from typing import Optional, List, Dict, Any
import openai

from agents.factories.agent_factory_multiagent import create_multiagent_system
from agents.core.context import prepare_manager_context, build_simple_manager_task


@tool
def reformulate_query(query: str, context_str: str, language: str = "fr") -> str:
    """
    Reformule une requête en tenant compte du contexte et de la langue cible.
    
    Args:
        query: La requête originale à reformuler
        context_str: Le contexte sous forme de chaîne de caractères
        language: La langue cible pour la reformulation (par défaut: "fr")
        
    Returns:
        La requête reformulée
    """
    return query  # L'agent smolagents s'occupera de la reformulation


class MultiAgentManager:
    """
    Multi-agent system following smolagents best practices.
    Manager delegates to specialized agents rather than doing work directly.
    """
    
    def __init__(self, model: OpenAIServerModel):
        """
        Initialize the multi-agent manager with a model.
        
        Args:
            model: OpenAI model to use for all agents
        """
        self.model = model
        self.manager_agent = None
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        self._initialized = False
        self.max_steps = 10  # Nombre maximum d'étapes à conserver
        
    def _manage_history(self, agent):
        """
        Gère l'historique de l'agent en limitant le nombre d'étapes.
        
        Args:
            agent: L'agent dont l'historique doit être géré
        """
        if len(agent.memory.steps) > self.max_steps:
            # Garder seulement les dernières étapes
            agent.memory.steps = agent.memory.steps[-self.max_steps:]
    
    def run_task(self, user_query: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a task through the multi-agent system.
        
        Args:
            user_query: The user's query
            additional_args: Additional arguments for context
            
        Returns:
            The result of the task
        """
        if not self._initialized:
            raise RuntimeError("Multi-agent system not initialized. Call initialize() first.")
        
        print(f"🎯 Processing task: {user_query}...")
        
        try:
            # Exécuter la tâche en conservant l'historique
            result = self.manager_agent.run(
                user_query,
                reset=False  # Conserver l'historique
            )
            
            # Gérer l'historique après l'exécution
            self._manage_history(self.manager_agent)
            
            print("✅ Task completed successfully!")
            
            return str(result)
            
        except Exception as e:
            print(f"❌ Error during task execution: {str(e)}")
            raise

    def reset_history(self):
        """
        Réinitialise l'historique de conversation.
        """
        if self.manager_agent:
            self.manager_agent.memory.steps = []
            print("🔄 Conversation history reset")

    def initialize(self):
        """Initialize the multi-agent system."""
        if self._initialized:
            print("⚠️ Multi-agent system already initialized")
            return
        
        print("🚀 Initializing multi-agent system...")
        
        # Create the multi-agent system
        (
            self.manager_agent,
            self.data_analyst_agent,
            self.document_agent,
            self.search_agent
        ) = create_multiagent_system(self.model)
        
        self._initialized = True
        print("✅ Multi-agent system initialized successfully!")
        print("   Architecture:")
        print("   ├── Manager Agent (minimal tools, coordinates everything)")
        print("   ├── Data Analyst Agent (CSV analysis & visualizations)")
        print("   ├── Document Agent (document processing)")
        print("   └── Search Agent (web research & information gathering)")

    def reset(self):
        """Reset the multi-agent system."""
        self._initialized = False
        self.manager_agent = None
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        print("🔄 Multi-agent system reset")


def initialize_multiagent_system(model: OpenAIServerModel) -> MultiAgentManager:
    """
    Initialize a new multi-agent system.
    
    Args:
        model: OpenAI model to use for all agents
        
    Returns:
        Initialized MultiAgentManager instance
    """
    manager = MultiAgentManager(model)
    manager.initialize()
    return manager


# Backward compatibility - alias for easy migration
def initialize_simplified_agent_system(model: OpenAIServerModel) -> MultiAgentManager:
    """
    Backward compatibility alias for existing code.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized MultiAgentManager (new architecture)
    """
    print("📢 Note: Migrating to new multi-agent architecture (smolagents best practices)")
    return initialize_multiagent_system(model) 