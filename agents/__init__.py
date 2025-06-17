"""
Clean Multi-Agent System for AlethéIA - Version Simplifiée
Architecture modulaire avec routage direct basé sur les mots-clés.
"""

# Multi-agent interface simplifié (recommandé)
from .agent_manager_multiagent import SimplifiedMultiAgentManager, initialize_multiagent_system

# Core utilities
from .core.embedding import get_embedding_function

# Backward compatibility
MultiAgentManager = SimplifiedMultiAgentManager

__all__ = [
    # Multi-agent interface simplifié (recommandé)
    'SimplifiedMultiAgentManager',
    'MultiAgentManager',  # Alias pour compatibilité
    'initialize_multiagent_system',
    
    # Core utilities
    'get_embedding_function'
] 