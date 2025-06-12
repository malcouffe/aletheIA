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
    
    Architecture:
    - Manager Agent: Pure delegation, no execution tools
    - Data Analyst Agent: CSV analysis & visualizations  
    - Document Agent: PDF search & analysis with unified tool
    - Search Agent: Web research & information gathering
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
    
    def initialize(self):
        """Initialize the multi-agent system."""
        if self._initialized:
            return
        
        print("🚀 Initializing multi-agent system (smolagents best practices)...")
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
    
    def run_task(self, user_query: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a task using the multi-agent system.
        Following smolagents best practices with minimal manager coordination.
        
        Args:
            user_query: The user's query/task
            additional_args: Optional context (PDF, CSV data, etc.)
            
        Returns:
            Response from the appropriate specialist agent
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")
        
        print(f"🎯 Processing task: {user_query[:100]}...")
        
        # Extract context data from additional_args for the context functions
        pdf_context = additional_args.get('pdf_context', {}) if additional_args else {}
        csv_context = additional_args.get('csv_context', {}) if additional_args else {}
        
        # Extract available_files lists for context preparation
        available_pdfs_context = pdf_context.get('available_files', []) if pdf_context else []
        available_csvs_context = csv_context.get('available_files', []) if csv_context else []
        
        # Prepare context using the correct format
        manager_context = prepare_manager_context(available_pdfs_context, available_csvs_context)
        manager_task = build_simple_manager_task(user_query, available_pdfs_context, available_csvs_context)
        
        # Execute via manager (which delegates to specialized agents)  
        result = self.manager_agent.run(
            task=manager_task,
            additional_args=additional_args or {}
        )
        
        print("✅ Task completed successfully!")
        return result
    
    def run_query(
        self,
        user_query: str,
        available_pdfs_context: Optional[List[Dict]] = None,
        available_csvs_context: Optional[List[Dict]] = None
    ) -> str:
        """
        Legacy method for backward compatibility.
        Converts old format to new smolagents-compliant format.
        
        Args:
            user_query: The user's question/request
            available_pdfs_context: List of PDF file context dictionaries
            available_csvs_context: List of CSV file context dictionaries
            
        Returns:
            Agent response as a string
        """
        if not self._initialized:
            self.initialize()
        
        # Convert legacy format to new additional_args format
        additional_args = {}
        
        if available_pdfs_context:
            additional_args['pdf_context'] = {
                'available_files': available_pdfs_context,
                'count': len(available_pdfs_context)
            }
            
        if available_csvs_context:
            additional_args['csv_context'] = {
                'available_files': available_csvs_context,
                'count': len(available_csvs_context)
            }
        
        # Use the new run_task method
        return self.run_task(user_query, additional_args)
    
    def get_embedding_function(self):
        """Get the shared embedding function."""
        return get_embedding_function()
    
    def reset(self):
        """Reset all agents for a fresh conversation."""
        if self._initialized:
            # Reset all agents if they have a reset method
            for agent in [self.manager_agent, self.data_analyst_agent, 
                         self.document_agent, self.search_agent]:
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
                "document_agent": {
                    "name": self.document_agent.name,
                    "tools_count": len(self.document_agent.tools)
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