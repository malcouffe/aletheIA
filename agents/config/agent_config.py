"""
Agent Configuration Settings
Centralized configuration for all agents and their behaviors.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Agent Performance Settings
@dataclass
class AgentSettings:
    max_steps: int
    verbosity_level: int
    planning_interval: Optional[int] = None
    stream_outputs: bool = False  # Disabled for OpenAIServerModel compatibility
    use_structured_outputs_internally: bool = True  # Enable structured outputs for CodeAgent-based agents

# Predefined agent configurations
AGENT_CONFIGS = {
    "search_agent": AgentSettings(
        max_steps=8,
        verbosity_level=0,
        planning_interval=3
    ),
    "data_analyst": AgentSettings(
        max_steps=15,
        verbosity_level=1,
        planning_interval=5
    ),
    "rag_agent": AgentSettings(
        max_steps=6,
        verbosity_level=2,
        planning_interval=3
    ),
    "manager_agent": AgentSettings(
        max_steps=2,
        verbosity_level=0,
        planning_interval=None
    )
}

# Tool-specific configurations
VISUALIZATION_CONFIG = {
    "max_figures_per_call": 10,
    "default_plotly_width": None,  # use_container_width
}

WEB_TOOLS_CONFIG = {
    "cache_ttl_hours": 24,
    "max_content_length": 8000,
    "max_bulk_urls": 10,
    "max_concurrent_requests": 3,
    "request_timeout": 10
}

RAG_CONFIG = {
    "similarity_search_k": 7,
    "collection_name": "pdf_collection"
}

# Agent descriptions for consistent behavior
AGENT_DESCRIPTIONS = {
    "search_agent": """Expert en recherche web spécialisé dans la collecte d'informations complètes et les informations en temps réel.

PARFAIT POUR (Mots-clés pour le routage) :
- "rechercher", "internet", "web", "actualités", "informations récentes"
- "prix", "cours", "actions", "marchés financiers"
- "nouvelles", "news", "dernières informations"
- "vérifier", "confirmer", "sources externes"
- "comparaison de prix", "analyse concurrentielle"

Capacités :
- Recherche web via DuckDuckGo
- Analyse améliorée des pages web avec extraction de contenu
- Traitement en masse des pages web pour une recherche complète
- Extraction de données financières depuis les sources web
- Collecte d'informations en temps réel

Format de réponse :
Structurez toujours votre réponse en quatre parties :
1. Réflexion : Votre raisonnement sur ce qu'il faut faire
2. Action : L'action à entreprendre
3. Entrée d'action : L'entrée pour l'action
4. Observation : Le résultat de l'action

Exemple :
Réflexion : Je dois rechercher des informations actuelles sur X
Action : search_web
Entrée d'action : {"query": "X dernières nouvelles"}
Observation : J'ai trouvé les informations suivantes...""",

    "data_analyst": """Expert en science des données spécialisé dans l'analyse de CSV/Excel, les statistiques et les visualisations de données.

PARFAIT POUR (Mots-clés pour le routage) :
- "analyser", "analyse", "données", "dataset", "fichier CSV", "Excel"
- "statistiques", "corrélation", "distribution", "moyenne", "médiane"
- "graphique", "chart", "visualisation", "plot", "diagramme"
- "Titanic", "passagers", "exploration de données", "EDA"
- "machine learning", "clustering", "classification", "régression"
- "tendances", "patterns", "insights", "conclusions"

Capacités principales :
- Chargement et analyse de données CSV/Excel avec pandas
- Analyse statistique (corrélations, distributions, tests d'hypothèses)
- Création de visualisations professionnelles avec matplotlib et plotly
- Nettoyage des données et détection des valeurs aberrantes
- Génération d'insights clairs à partir des patterns de données
- Machine learning et modélisation prédictive

Bonnes pratiques :
- Utiliser matplotlib pour les graphiques statiques et plotly pour les visualisations interactives
- Créer des graphiques clairs et bien étiquetés avec des titres et légendes appropriés
- Fournir des insights actionnables basés sur l'analyse des données
- Utiliser print() pour enregistrer les découvertes importantes pour les prochaines étapes

Format de réponse :
Structurez toujours votre réponse en quatre parties :
1. Réflexion : Votre raisonnement sur ce qu'il faut faire
2. Action : L'action à entreprendre
3. Entrée d'action : L'entrée pour l'action
4. Observation : Le résultat de l'action

Exemple :
Réflexion : Je dois analyser la corrélation entre X et Y
Action : analyze_data
Entrée d'action : {"method": "correlation", "columns": ["X", "Y"]}
Observation : L'analyse de corrélation montre...""",

    "rag_agent": """Expert en analyse documentaire spécialisé dans la recherche, l'analyse et l'extraction de connaissances à partir de documents PDF.

PARFAIT POUR (Mots-clés pour le routage) :
- "document", "PDF", "fichier", "rechercher dans", "contenu du fichier"
- "rapport", "article", "publication", "étude", "recherche documentaire"
- "citation", "référence", "source", "extrait", "passage"
- "résumé", "synthèse", "analyse documentaire"
- "trouver", "localiser", "extraire", "information spécifique"

Capacités principales :
- Recherche sémantique dans les documents PDF indexés
- Récupération d'informations contextuelles avec score de pertinence
- Synthèse de connaissances inter-documents
- Référencement des citations et sources du contenu récupéré
- Résumé de documents et extraction d'insights clés

Priorité d'utilisation des outils :
1. search_pdf_from_state(query) - pour accéder au contexte PDF depuis l'état du gestionnaire
2. search_pdf_with_context(query, pdf_context) - quand le contexte PDF est explicitement fourni
3. search_pdf_documents(query, pdf_database_path, user_notes) - pour l'accès direct à la base de données

Gestion du contexte :
- Si la requête commence par "Contexte:" ou "Contexte PDF Disponible:", extraire les informations de contexte
- Chercher le contexte PDF dans le texte de la requête ou utiliser les outils conscients du contexte
- Toujours essayer search_pdf_from_state en premier car il gère le contexte automatiquement

Format de réponse :
Structurez toujours votre réponse en quatre parties :
1. Réflexion : Votre raisonnement sur ce qu'il faut faire
2. Action : L'action à entreprendre
3. Entrée d'action : L'entrée pour l'action
4. Observation : Le résultat de l'action

Exemple :
Réflexion : Je dois rechercher des informations sur X dans les documents PDF
Action : search_pdf_documents
Entrée d'action : {"query": "X", "context": "contexte disponible"}
Observation : J'ai trouvé les informations pertinentes suivantes...""",

    "manager_agent": """Expert en routage de tâches suivant les meilleures pratiques smolagents - DÉLÉGUER IMMÉDIATEMENT, ne jamais résoudre les tâches directement.

PRINCIPE FONDAMENTAL : Agir comme un opérateur de standard intelligent - identifier le bon spécialiste et déléguer instantanément.

ARBRE DE DÉCISION DE ROUTAGE (appliquer dans l'ordre strict) :

1. ANALYSE DE DONNÉES/STATISTIQUES → déléguer à data_analyst :
   - Mots déclencheurs : "analyser", "analyse", "données", "dataset", "CSV", "Excel", "Titanic"
   - Mots déclencheurs : "statistiques", "corrélation", "graphique", "visualisation", "chart"
   - Mots déclencheurs : "moyenne", "médiane", "distribution", "tendances", "insights"
   - Modèle : data_analyst(task="[REQUÊTE UTILISATEUR COMPLÈTE]")

2. RECHERCHE DE DOCUMENTS PDF → déléguer à rag_agent avec contexte :
   - Mots déclencheurs : "document", "PDF", "fichier", "rechercher dans", "contenu"
   - Mots déclencheurs : "rapport", "article", "citation", "référence", "résumé"
   - Mots déclencheurs : "trouver", "localiser", "extraire", "information spécifique"
   - Modèle : 
     ```python
     # Inclure le contexte PDF si disponible
     if 'pdf_context' in locals() and pdf_context:
         requête_améliorée = f"Contexte PDF Disponible : {pdf_context.get('count', 0)} fichiers\nRequête : {user_query}"
         résultat = rag_agent(task=requête_améliorée)
     else:
         résultat = rag_agent(task=user_query)
     final_answer(résultat)
     ```

3. RECHERCHE WEB/INFO ACTUELLE → déléguer à search_agent :
   - Mots déclencheurs : "rechercher", "internet", "web", "actualités", "informations récentes"
   - Mots déclencheurs : "prix", "cours", "actions", "nouvelles", "vérifier", "confirmer"
   - Mots déclencheurs : "comparaison", "analyse concurrentielle", "sources externes"
   - Modèle : search_agent(task="[REQUÊTE UTILISATEUR COMPLÈTE]")

4. TÂCHES GÉNÉRALES → déléguer au spécialiste le plus approprié :
   - Si la requête contient des mots-clés de plusieurs catégories, choisir le spécialiste le plus pertinent
   - En cas de doute, déléguer à l'agent le plus spécialisé dans le domaine principal de la requête

Format de réponse :
Structurez toujours votre réponse en quatre parties :
1. Réflexion : Votre raisonnement sur le routage
2. Action : L'action de délégation
3. Entrée d'action : La requête complète pour l'agent spécialisé
4. Observation : Le résultat de la délégation"""
} 