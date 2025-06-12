"""
Clean Multi-Agent System for AlethéIA
Modular architecture with separation of concerns.
"""

# Multi-agent interface following smolagents best practices (recommended)
from .agent_manager_multiagent import MultiAgentManager, initialize_multiagent_system

# Core utilities
from .core.embedding import get_embedding_function

__all__ = [
    # Multi-agent interface (recommended)
    'MultiAgentManager',
    'initialize_multiagent_system',
    
    # Core utilities
    'get_embedding_function'
] 