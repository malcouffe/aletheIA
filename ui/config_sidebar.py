"""
Configuration sidebar for API keys and file upload.
"""
import streamlit as st
import os
from config.settings import OPENAI_API_KEY_ENV, MISTRAL_API_KEY_ENV, SUPPORTED_FILE_TYPES


def display_config_sidebar():
    """Display the configuration sidebar with API keys and file upload."""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key configuration
        api_key, mistral_api_key = _configure_api_keys()
        
        # File upload section
        st.subheader("Chargement de Fichier")
        uploaded_file = st.file_uploader(
            "Charger un fichier (PDF, CSV, TXT)",
            type=SUPPORTED_FILE_TYPES,
            key="unified_uploader"
        )
        
        return api_key, mistral_api_key, uploaded_file


def _configure_api_keys():
    """Configure and display API key inputs."""
    # OpenAI API Key
    api_key_from_env = os.environ.get(OPENAI_API_KEY_ENV)
    api_key = api_key_from_env
    
    if not api_key_from_env:
        api_key = st.text_input("Clé API OpenAI", type="password", key="openai_api_key_input")
    else:
        if not st.session_state.get('openai_key_toast_shown', False):
            st.toast("Clé API OpenAI (env) trouvée", icon="✅")
            st.session_state.openai_key_toast_shown = True
    
    # Mistral API Key
    mistral_api_key_from_env = os.environ.get(MISTRAL_API_KEY_ENV)
    mistral_api_key = mistral_api_key_from_env
    
    if not mistral_api_key_from_env:
        mistral_api_key = st.text_input("Clé API Mistral (pour PDF)", type="password", key="mistral_api_key_input")
    else:
        if not st.session_state.get('mistral_key_toast_shown', False):
            st.toast("Clé API Mistral (env) trouvée", icon="✅")
            st.session_state.mistral_key_toast_shown = True
    
    return api_key, mistral_api_key 