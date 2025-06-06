"""
Embedding Functions for RAG and Vector Search
Centralized embedding management for consistent behavior across agents.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.agent_config import EMBEDDING_MODEL_NAME

# Global embedding function instance for reuse
_embedding_function = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Gets or initializes the global embedding function instance.
    Reuses the same instance for efficiency.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding function
    """
    global _embedding_function
    
    if _embedding_function is None:
        _embedding_function = initialize_embedding_function()
    
    return _embedding_function


def initialize_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initializes and returns a new HuggingFace embedding function.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding function
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def reset_embedding_function():
    """
    Resets the global embedding function instance.
    Useful for testing or configuration changes.
    """
    global _embedding_function
    _embedding_function = None 