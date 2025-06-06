"""
Configuration settings for the aletheIA application.
"""
import os

# Directory paths
PERSISTENCE_DIR = os.path.join("data", "session_persistence")
PERSISTENCE_FILE = os.path.join(PERSISTENCE_DIR, "persistent_session_state.json")
TEMP_FILES_DIR = os.path.join("data", "temp_files")
VECTOR_DB_DIR = os.path.join("data", "vector_db")
FIGURES_DIR = os.path.join("data", "figures")

# File processing settings
SUPPORTED_FILE_TYPES = ['pdf', 'csv', 'txt']

# Status mappings
STATUS_MAP = {
    'awaiting_classification': "en attente de classification",
    'classified': "classifié, prêt pour indexation",
    'indexed': "indexé et prêt pour questions",
    'ready': "prêt pour analyse",
    'error_extraction': "erreur d'extraction de texte",
    'error_analysis': "erreur d'analyse IA",
    'error_validation': "erreur de validation CSV",
    'error_indexing': "erreur d'indexation",
    'error_indexing_missing_temp': "Erreur Indexation (Fichier temporaire manquant)",
    'unknown': "statut inconnu"
}

# UI settings
FILE_TYPE_ICONS = {
    'pdf': "📄",
    'csv': "📊",
    'unknown': "❓"
}

# Environment variables
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY" 