"""
Simplified Agent Manager Interface
Manages only Enhanced Manager Agent + Data Analyst Agent.
"""

from smolagents import OpenAIServerModel
from typing import Optional, List, Dict, Any

from .factories.agent_factory_simplified import create_simplified_agent_system
from .core.context import prepare_manager_context, build_simple_manager_task
from .core.embedding import get_embedding_function


class SimplifiedAgentManager:
    """
    Simplified interface for managing the streamlined two-agent system.
    Enhanced Manager handles web search and RAG directly, delegates data analysis.
    """
    
    def __init__(self, model: OpenAIServerModel):
        """
        Initialize the simplified agent manager with a model.
        
        Args:
            model: OpenAI model to use for all agents
        """
        self.model = model
        self.enhanced_manager_agent = None
        self.data_analyst_agent = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the simplified agent system."""
        if self._initialized:
            return
        
        print("🚀 Initializing simplified agent system...")
        (
            self.enhanced_manager_agent,
            self.data_analyst_agent
        ) = create_simplified_agent_system(self.model)
        
        self._initialized = True
        print("✅ Simplified agent system initialized successfully!")
        print(f"   - Enhanced Manager: All-in-one (web + RAG + delegation)")
        print(f"   - Data Analyst: Specialized for data analysis")
    
    def run_query(
        self,
        user_query: str,
        available_pdfs_context: Optional[List[Dict]] = None,
        available_csvs_context: Optional[List[Dict]] = None
    ) -> str:
        """
        Run a user query through the enhanced manager agent.
        
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
        

        
        # Run enhanced manager with context - smolagents makes additional_args available as state variables
        return self.enhanced_manager_agent.run(
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
            for agent in [self.enhanced_manager_agent, self.data_analyst_agent]:
                if agent and hasattr(agent, 'reset'):
                    agent.reset()

def initialize_simplified_agent_system(model: OpenAIServerModel) -> SimplifiedAgentManager:
    """
    Initialize and return a SimplifiedAgentManager instance.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized SimplifiedAgentManager
    """
    manager = SimplifiedAgentManager(model)
    manager.initialize()
    return manager