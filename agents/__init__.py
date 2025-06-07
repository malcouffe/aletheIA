"""
Clean Multi-Agent System for AlethéIA
Modular architecture with separation of concerns.
"""

# Simplified interface - streamlined architecture
from .agent_manager_simplified import SimplifiedAgentManager, initialize_simplified_agent_system

# Core utilities
from .core.embedding import get_embedding_function

__all__ = [
    # Simplified interface
    'SimplifiedAgentManager',
    'initialize_simplified_agent_system',
    
    # Core utilities
    'get_embedding_function'
] 