"""
State synchronization utilities to keep UI components in sync.
"""
import streamlit as st
import time
from core.session_manager import save_persistent_state


def sync_file_status(file_id: str, new_status: str, message: str = None, force_rerun: bool = True):
    """
    Synchronize file status across all UI components.
    
    Args:
        file_id: The file ID to update
        new_status: The new status to set
        message: Optional message to add to chat
        force_rerun: Whether to force a Streamlit rerun
    """
    if file_id not in st.session_state.processed_files:
        return False
    
    # Update the file status
    st.session_state.processed_files[file_id]['status'] = new_status
    
    # Save to persistent storage
    save_persistent_state(st.session_state.processed_files)
    
    # Add chat message if provided
    if message:
        st.session_state.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": time.time()
        })
    
    # Force UI update if requested
    if force_rerun:
        st.rerun()
    
    return True


def get_status_emoji(status: str) -> str:
    """Get the appropriate emoji for a given status."""
    status_emojis = {
        'awaiting_classification': '⏳',
        'classified': '📝',
        'indexing': '🔄',
        'indexed': '✅',
        'ready': '✅',
        'error_extraction': '❌',
        'error_analysis': '❌',
        'error_validation': '❌',
        'error_indexing': '❌',
        'error_indexing_missing_temp': '❌',
        'unknown': '❓'
    }
    return status_emojis.get(status, '❓')


def get_status_message(status: str, filename: str) -> str:
    """Generate a user-friendly status message."""
    status_messages = {
        'awaiting_classification': f"📄 '{filename}' est en attente de classification",
        'classified': f"📝 '{filename}' est classifié et prêt pour l'indexation",
        'indexing': f"🔄 Indexation de '{filename}' en cours...",
        'indexed': f"✅ '{filename}' a été indexé avec succès et est prêt pour les questions",
        'ready': f"✅ '{filename}' est prêt pour l'analyse",
        'error_extraction': f"❌ Erreur lors de l'extraction de texte de '{filename}'",
        'error_analysis': f"❌ Erreur lors de l'analyse IA de '{filename}'",
        'error_validation': f"❌ Erreur de validation pour '{filename}'",
        'error_indexing': f"❌ Erreur lors de l'indexation de '{filename}'",
        'error_indexing_missing_temp': f"❌ Erreur d'indexation de '{filename}' (fichier temporaire manquant)",
        'unknown': f"❓ Statut inconnu pour '{filename}'"
    }
    return status_messages.get(status, f"Statut '{status}' pour '{filename}'")


def sync_chat_with_status(file_id: str):
    """
    Synchronize chat messages with the current file status.
    This is now only used internally by the sync system.
    """
    if file_id not in st.session_state.processed_files:
        return
    
    file_details = st.session_state.processed_files[file_id]
    filename = file_details.get('filename', f'ID: {file_id}')
    current_status = file_details.get('status', 'unknown')
    
    # Add status message to chat only if it's an important transition
    important_statuses = ['indexed', 'ready', 'error_indexing', 'error_analysis']
    if current_status in important_statuses:
        new_message = get_status_message(current_status, filename)
        st.session_state.messages.append({
            "role": "assistant",
            "content": new_message,
            "timestamp": time.time()
        })


def is_status_transition_valid(old_status: str, new_status: str) -> bool:
    """Check if a status transition is valid."""
    valid_transitions = {
        'awaiting_classification': ['classified', 'error_analysis'],
        'classified': ['indexing', 'error_indexing', 'indexed'],
        'indexing': ['indexed', 'error_indexing'],
        'indexed': ['indexing'],  # Allow re-indexing
        'ready': ['error_analysis'],
        'error_extraction': ['awaiting_classification'],
        'error_analysis': ['awaiting_classification'],
        'error_validation': ['ready'],
        'error_indexing': ['classified', 'indexing'],
        'error_indexing_missing_temp': ['classified'],
    }
    
    return new_status in valid_transitions.get(old_status, []) 