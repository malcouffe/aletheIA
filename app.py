"""
Main Streamlit application for aletheIA document analysis assistant.
"""
import streamlit as st
import uuid
import torch
import traceback
import os
import logging
from smolagents import OpenAIServerModel

# Configure logging to suppress HTTP/API debug logs but keep RAG functionality visible
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING) 
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("smolagents").setLevel(logging.WARNING)

# Import from new modular structure
from config.settings import SUPPORTED_FILE_TYPES
from core.session_manager import load_persistent_state, save_persistent_state
from ui.config_sidebar import display_config_sidebar
from ui.sidebar import display_processed_files_sidebar
from ui.pdf_actions import display_pdf_action_section
from ui.chat import display_chat_interface
from ui.components import handle_uploaded_file
from ui.state_sync import sync_file_status
from ui.realtime_notifications import create_status_toast


# Import from multi-agent architecture
from agents import MultiAgentManager
import time

# Prevent torch serialization issues
torch.classes.__path__ = []


def initialize_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        if 'processed_files' not in st.session_state:
             st.session_state.processed_files = load_persistent_state()
        st.session_state.selected_file_id_for_action = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def initialize_agents(api_key):
    """Initialize the agent manager with multi-agent architecture."""
    model = None
    agent_manager = None

    if api_key:
        try:
            if 'model' not in st.session_state or st.session_state.model is None:
                 st.session_state.model = OpenAIServerModel(
                    model_id="gpt-4o-mini",
                    api_base="https://api.openai.com/v1",
                    api_key=api_key,
                )
            model = st.session_state.model

            # Initialize multi-agent manager
            if 'agent_manager' not in st.session_state or st.session_state.agent_manager is None:
                st.session_state.agent_manager = MultiAgentManager(model)
                st.session_state.agent_manager.initialize()
            
            agent_manager = st.session_state.agent_manager

        except Exception as e:
            st.error(f"Erreur d'initialisation du modèle ou des agents OpenAI : {e}")
            st.error(traceback.format_exc())
            model = None
    else:
        st.warning("Veuillez entrer votre clé API OpenAI dans la barre latérale pour activer les fonctionnalités d'IA.")

    return model, agent_manager


def handle_file_upload(uploaded_file, api_key, mistral_api_key, model):
    """Handle file upload and processing."""
    if uploaded_file is not None:
        if not api_key or (uploaded_file.type == "application/pdf" and not mistral_api_key):
            st.error("Veuillez fournir les clés API nécessaires (OpenAI et/ou Mistral pour les PDF) avant de charger des fichiers.")
            return

        # NO initial notification - keep it clean
        with st.spinner(f"Traitement de {uploaded_file.name}..."):
            processed_file_info = handle_uploaded_file(
                uploaded_file=uploaded_file,
                session_id=st.session_state.session_id,
                model=model,
                mistral_api_key=mistral_api_key,
                use_memory_limit=True,
            )

        _process_upload_result(uploaded_file, processed_file_info)


def _process_upload_result(uploaded_file, processed_file_info):
    """Process the result of file upload with minimal notifications."""
    if processed_file_info and 'already_processed' in processed_file_info:
        # Simple info message, no chat spam
        st.info(f"Fichier '{uploaded_file.name}' déjà traité dans cette session.")
        
    elif processed_file_info and 'file_id' in processed_file_info and 'details' in processed_file_info:
        file_id = processed_file_info['file_id']
        details = processed_file_info['details']
        
        # Update processed files
        st.session_state.processed_files[file_id] = details
        save_persistent_state(st.session_state.processed_files)

        # Add ONE concise success notification to chat
        _add_concise_success_notification(details)
        
        st.rerun() 

    elif processed_file_info is False:
        error_message = f"❌ Échec du traitement de '{uploaded_file.name}' : clé API manquante."
        st.session_state.messages.append({
            "role": "assistant", 
            "content": error_message,
            "timestamp": time.time()
        })
        st.error(error_message)
    else:
        warning_message = f"⚠️ Échec du traitement de '{uploaded_file.name}'."
        st.session_state.messages.append({
            "role": "assistant", 
            "content": warning_message,
            "timestamp": time.time()
        })
        st.warning(warning_message)


def _add_concise_success_notification(details):
    """Add a single, concise success notification to the chat."""
    filename = details.get('filename', 'Inconnu')
    file_type_display = details.get('type', 'unknown').upper()
    status = details.get('status', 'unknown')
    
    # Create ONE simple success message
    if file_type_display == 'PDF':
        if status == 'awaiting_classification':
            message = f"✅ **{filename}** ajouté. Classification et indexation disponibles dans la barre latérale."
        else:
            message = f"✅ **{filename}** traité avec succès."
            
    elif file_type_display == 'CSV':
        rows = details.get('rows', 'N/A')
        cols = len(details.get('columns', []))
        message = f"✅ **{filename}** prêt. {rows} lignes, {cols} colonnes. Vous pouvez poser vos questions !"
    else:
        message = f"✅ **{filename}** traité avec succès."
    
    # Add the message to chat
    st.session_state.messages.append({
        "role": "assistant", 
        "content": message,
        "timestamp": time.time()
    })


def main():
    """Main application function."""
    st.set_page_config(layout="wide")
    st.title("Assistant d'Analyse de Documents (PDF/CSV)")
    st.write("Chargez un fichier PDF ou CSV. Configurez vos clés API dans la barre latérale si nécessaire.")

    # Initialize session state
    initialize_session_state()

    # Display configuration sidebar and get inputs
    api_key, mistral_api_key, uploaded_file = display_config_sidebar()

    # Initialize agents with multi-agent architecture
    model, agent_manager = initialize_agents(api_key)

    # Handle file upload
    handle_file_upload(uploaded_file, api_key, mistral_api_key, model)

    # Display PDF action section
    display_pdf_action_section(mistral_api_key)

    # Display processed files sidebar
    display_processed_files_sidebar()

    # Display chat interface
    display_chat_interface(model, agent_manager)
    
    # Add a section to display any existing charts
    _display_existing_charts()


def _display_existing_charts():
    """Display any existing charts that were created by the data analyst."""
    if 'session_id' not in st.session_state:
        return
    
    # Look for charts in the figures directory
    figures_base = os.path.join('data', 'figures', st.session_state.session_id)
    
    if os.path.exists(figures_base):
        # Find all chart files
        chart_paths = []
        for root, dirs, files in os.walk(figures_base):
            for file in files:
                if file.endswith(('.png', '.jpg', '.svg')):
                    chart_paths.append(os.path.join(root, file))
        
        if chart_paths:
            # Sort by modification time (newest first)
            chart_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Only show recent charts (created in the last hour)
            current_time = time.time()
            recent_charts = [path for path in chart_paths if current_time - os.path.getmtime(path) < 3600]
            
            if recent_charts:
                st.divider()
                st.subheader("📊 Recently Generated Charts")
                
                # Display in columns
                cols = st.columns(min(len(recent_charts), 3))
                
                for i, chart_path in enumerate(recent_charts):
                    chart_name = os.path.basename(chart_path)
                    display_name = chart_name.replace('.png', '').replace('_', ' ').title()
                    
                    with cols[i % 3]:
                        try:
                            st.image(chart_path, caption=display_name, use_column_width=True)
                            
                            # Add timestamp
                            mod_time = os.path.getmtime(chart_path)
                            import datetime
                            mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%H:%M:%S")
                            st.caption(f"🕒 Created at {mod_time_str}")
                            
                        except Exception as e:
                            st.error(f"Error displaying {chart_name}: {e}")


if __name__ == "__main__":
    main() 