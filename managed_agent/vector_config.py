"""
Configuration centralisée pour les bases de données vectorielles.
"""

import os
from typing import Dict, Any

# Paramètres de l'embedding model
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_kwargs": {'device': 'cpu'},
    "encode_kwargs": {'normalize_embeddings': False}
}

# Paramètres de chunking pour l'indexation
CHUNKING_CONFIG = {
    "chunk_size": 1800,  # Taille des chunks (augmentée de 1000 à 1800)
    "chunk_overlap": 300,  # Chevauchement (augmenté de 200 à 300)
    # Séparateurs pour le découpage du texte
    "separators": [
        "\n\n",       # Double saut de ligne
        "\n",         # Saut de ligne simple
        ". ",         # Fin de phrase
        "! ",         # Fin de phrase exclamative
        "? ",         # Fin de phrase interrogative
        ":",          # Deux-points
        ";",          # Point-virgule
        ", ",         # Virgule
        " "           # Espace (dernier recours)
    ]
}

# Paramètres de recherche
SEARCH_CONFIG = {
    "k_value": 5,              # Nombre de documents à retourner dans les recherches standard
    "mmr_k": 5,                # Nombre de documents à retourner dans les recherches MMR
    "mmr_fetch_k": 10,         # Nombre de documents à récupérer avant de filtrer avec MMR
    "mmr_lambda_mult": 0.5,    # Facteur de diversité pour MMR (0.5 = équilibre entre pertinence et diversité)
    "max_docs": 7,             # Nombre maximum de documents à retourner au total
}

# Indicateurs de définition pour le scoring de relevance
DEFINITION_INDICATORS = [
    "are defined as", 
    "refers to", 
    "is defined as", 
    "can be defined as", 
    "means", 
    "represent",
    "definition of", 
    "concept of"
]

# Mots-clés pour détecter les demandes de définition
DEFINITION_KEYWORDS = [
    "définition", 
    "definition", 
    "concept", 
    "meaning", 
    "what is", 
    "qu'est-ce que", 
    "signification"
]

# Préfixes de requête à nettoyer pour les définitions
DEFINITION_PREFIXES = [
    "définition de", 
    "definition of", 
    "concept de", 
    "concept of", 
    "what is", 
    "qu'est-ce que"
]

# Termes techniques spécifiques et leurs variations pour la recherche
TECHNICAL_TERMS = {
    "digital_finance": [
        "digital finance", 
        "finance numérique", 
        "digital finance definition",
        "finance numérique définition",
        "digital finance concept",
        "digital finance refers to"
    ],
    "regulation": [
        "regulation", 
        "regulatory framework", 
        "réglementation",
        "cadre réglementaire"
    ],
    "blockchain": [
        "blockchain", 
        "chain of blocks", 
        "distributed ledger"
    ]
}

# Requêtes de secours à utiliser si aucun résultat n'est trouvé
BACKUP_QUERIES = [
    "finance", 
    "banking", 
    "information", 
    "key concepts"
]

def get_db_path(classification: str, session_id: str = None) -> str:
    """
    Construit le chemin de la base de données vectorielle.
    
    Args:
        classification: La classification du document (ex: 'Tokenisation')
        session_id: Identifiant de session optionnel
    
    Returns:
        Le chemin vers le répertoire de la base de données
    """
    base_path = os.path.join("data", "output", "vectordb", classification)
    if session_id:
        return os.path.join(base_path, f"session_{session_id}")
    return base_path 