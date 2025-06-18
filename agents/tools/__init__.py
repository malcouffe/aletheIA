"""
Data Analysis Tools - OPTIMISÉ SELON LES BONNES PRATIQUES SMOLAGENTS
Outil unifié principal pour réduire la complexité et améliorer la performance.
"""

from .unified_data_tools import load_and_explore_csv, display_figures, data_loader, load_csv_data, get_dataframe
from .visualization_tools import check_undisplayed_figures  # Keep this for backward compatibility
from .rag_tools import rag_search_simple, get_latest_rag_sources, clear_rag_sources_cache

__all__ = [
    # OUTIL PRINCIPAL - BONNES PRATIQUES SMOLAGENTS
    'load_and_explore_csv',  # OUTIL UNIFIÉ - Combine chargement, découverte et exploration
    'display_figures',       # Affichage des graphiques avec débogage détaillé
    
    # OUTILS LEGACY - COMPATIBILITÉ ASCENDANTE (DÉPRÉCIÉS)
    'load_csv_data',         # → Redirige vers load_and_explore_csv
    'data_loader',          # → Redirige vers load_and_explore_csv  
    'get_dataframe',        # → Redirige vers load_and_explore_csv
    
    # AUTRES OUTILS
    'check_undisplayed_figures',
    'rag_search_simple',
    'get_latest_rag_sources',
    'clear_rag_sources_cache'
] 