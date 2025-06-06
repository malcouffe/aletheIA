"""
Clean Multi-Agent System for AlethéIA
Modular architecture with separation of concerns.
"""

# Main interface - use this for new code
from .agent_manager import AgentManager, initialize_agent_system

# Backward compatibility - keep these for existing code that imports from agents.utils
from .agent_manager import run_manager_with_additional_args
from .core.embedding import get_embedding_function

__all__ = [
    # New clean interface
    'AgentManager',
    'initialize_agent_system',
    
    # Backward compatibility
    'run_manager_with_additional_args',
    'get_embedding_function'
] 