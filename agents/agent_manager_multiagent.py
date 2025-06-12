"""
Multi-Agent Manager Interface
Implements smolagents best practices with specialized agents and minimal manager.
"""

from smolagents import OpenAIServerModel
from typing import Optional, List, Dict, Any

from .factories.agent_factory_multiagent import create_multiagent_system
from .core.context import prepare_manager_context, build_simple_manager_task
from .core.embedding import get_embedding_function


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
        self.rag_agent = None
        self.search_agent = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the multi-agent system."""
        if self._initialized:
            return
        
        print("🚀 Initializing multi-agent system (smolagents best practices)...")
        (
            self.manager_agent,
            self.data_analyst_agent,
            self.rag_agent,
            self.search_agent
        ) = create_multiagent_system(self.model)
        
        self._initialized = True
        print("✅ Multi-agent system initialized successfully!")
        print("   Architecture:")
        print("   ├── Manager Agent (minimal tools, coordinates everything)")
        print("   ├── Data Analyst Agent (CSV analysis & visualizations)")
        print("   ├── RAG Agent (PDF document search & analysis)")
        print("   └── Search Agent (web research & information gathering)")
    
    def run_query(
        self,
        user_query: str,
        available_pdfs_context: Optional[List[Dict]] = None,
        available_csvs_context: Optional[List[Dict]] = None
    ) -> str:
        """
        Run a user query through the multi-agent system.
        
        Args:
            user_query: The user's question/request
            available_pdfs_context: List of PDF file context dictionaries
            available_csvs_context: List of CSV file context dictionaries
            
        Returns:
            Agent response as a string
        """
        if not self._initialized:
            self.initialize()
        
        # Build task description and context properly
        task = build_simple_manager_task(
            user_query, 
            available_pdfs_context, 
            available_csvs_context
        )
        
        context = prepare_manager_context(
            available_pdfs_context, 
            available_csvs_context
        )
        
        # The manager will delegate to appropriate specialized agents with proper context
        return self.manager_agent.run(
            task=task,
            additional_args=context,  # Pass the actual context instead of empty dict
            reset=True  # Fresh start for each query
        )
    
    def get_embedding_function(self):
        """Get the shared embedding function."""
        return get_embedding_function()
    
    def reset(self):
        """Reset all agents for a fresh conversation."""
        if self._initialized:
            # Reset all agents if they have a reset method
            for agent in [self.manager_agent, self.data_analyst_agent, 
                         self.rag_agent, self.search_agent]:
                if agent and hasattr(agent, 'reset'):
                    agent.reset()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about all agents."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "manager": {
                "name": self.manager_agent.name,
                "tools_count": len(self.manager_agent.tools),
                "managed_agents_count": len(self.manager_agent.managed_agents)
            },
            "specialists": {
                "data_analyst": {
                    "name": self.data_analyst_agent.name,
                    "tools_count": len(self.data_analyst_agent.tools)
                },
                "rag_agent": {
                    "name": self.rag_agent.name,
                    "tools_count": len(self.rag_agent.tools)
                },
                "search_agent": {
                    "name": self.search_agent.name,
                    "tools_count": len(self.search_agent.tools)
                }
            },
            "architecture": "smolagents_compliant"
        }


def initialize_multiagent_system(model: OpenAIServerModel) -> MultiAgentManager:
    """
    Initialize and return a MultiAgentManager instance.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized MultiAgentManager
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