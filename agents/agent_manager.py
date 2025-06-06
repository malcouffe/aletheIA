"""
Clean Agent Manager Interface
Replaces the monolithic utils.py with a simple, clean interface.
"""

from smolagents import OpenAIServerModel
from typing import Optional, List, Dict, Any

from .factories.agent_factory import create_agent_system
from .core.context import prepare_manager_context, build_simple_manager_task
from .core.embedding import get_embedding_function


class AgentManager:
    """
    Main interface for managing the multi-agent system.
    Provides a clean, simple API for agent interactions.
    """
    
    def __init__(self, model: OpenAIServerModel):
        """
        Initialize the agent manager with a model.
        
        Args:
            model: OpenAI model to use for all agents
        """
        self.model = model
        self.manager_agent = None
        self.search_agent = None
        self.data_analyst_agent = None
        self.rag_agent = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all agents in the system."""
        if self._initialized:
            return
        
        print("🚀 Initializing agent system...")
        (
            self.manager_agent,
            self.search_agent, 
            self.data_analyst_agent,
            self.rag_agent
        ) = create_agent_system(self.model)
        
        self._initialized = True
        print("✅ Agent system initialized successfully!")
    
    def run_query(
        self,
        user_query: str,
        available_pdfs_context: Optional[List[Dict]] = None,
        available_csvs_context: Optional[List[Dict]] = None
    ) -> str:
        """
        Run a user query through the appropriate agent(s).
        
        Args:
            user_query: The user's question/request
            available_pdfs_context: List of PDF file context dictionaries
            available_csvs_context: List of CSV file context dictionaries
            
        Returns:
            Agent response as a string
        """
        if not self._initialized:
            self.initialize()
        
        # Build task description and context
        task = build_simple_manager_task(
            user_query, 
            available_pdfs_context, 
            available_csvs_context
        )
        
        context = prepare_manager_context(
            available_pdfs_context, 
            available_csvs_context
        )
        
        print(f"🔍 DEBUG run_query:")
        print(f"  - Context keys: {list(context.keys())}")
        print(f"  - Passing context as additional_args to agent.run()")
        
        # Run with context in additional_args - smolagents will handle making these available as state variables
        return self.manager_agent.run(
            task=task,
            additional_args=context,
            reset=True  # Fresh start for each query
        )
    
    def get_embedding_function(self):
        """Get the shared embedding function."""
        return get_embedding_function()
    
    def reset(self):
        """Reset all agents for a fresh conversation."""
        if self._initialized:
            # Reset agents if they have a reset method
            for agent in [self.manager_agent, self.search_agent, self.data_analyst_agent, self.rag_agent]:
                if agent and hasattr(agent, 'reset'):
                    agent.reset()


# Convenience functions for backward compatibility with existing code

def initialize_agent_system(model: OpenAIServerModel) -> AgentManager:
    """
    Initialize and return an AgentManager instance.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized AgentManager
    """
    manager = AgentManager(model)
    manager.initialize()
    return manager


def run_manager_with_additional_args(
    manager_agent,
    user_query: str,
    available_pdfs_context: Optional[List[Dict]] = None,
    available_csvs_context: Optional[List[Dict]] = None
):
    """
    Backward compatibility function for existing code.
    Use AgentManager.run_query() instead for new code.
    """
    # Build task description and context using the new modules
    task = build_simple_manager_task(
        user_query, 
        available_pdfs_context, 
        available_csvs_context
    )
    
    context = prepare_manager_context(
        available_pdfs_context, 
        available_csvs_context
    )
    
    print(f"🔍 DEBUG run_manager_with_additional_args:")
    print(f"  - Context keys: {list(context.keys())}")
    print(f"  - Passing context as additional_args to agent.run()")
    
    # Run with context in additional_args - smolagents will handle making these available as state variables
    return manager_agent.run(
        task=task,
        additional_args=context,
        reset=True
    ) 