"""
PDF action UI components for classification and indexing.
"""
import streamlit as st
from ui.components import index_pdf, PDF_CLASSES
from ui.state_sync import sync_file_status, sync_chat_with_status
from core.session_manager import save_persistent_state, update_user_notes_callback, update_classification_callback
import time


def display_pdf_action_section(mistral_api_key: str):
    """Displays the UI section for PDF classification and indexing for a selected PDF."""
    selected_file_id = st.session_state.get('selected_file_id_for_action')

    if not selected_file_id or selected_file_id not in st.session_state.get('processed_files', {}):
        return

    file_details = st.session_state.processed_files[selected_file_id]

    if file_details.get('type') != 'pdf':
        st.warning("Le fichier sélectionné n'est pas un PDF.")
        return

    st.subheader(f"Classification/Indexation : {file_details.get('filename', selected_file_id)}")
    
    _display_pdf_summary(file_details)
    _display_user_notes_section(selected_file_id, file_details)
    _display_classification_section(selected_file_id, file_details)
    _display_indexing_section(selected_file_id, file_details, mistral_api_key)
    
    st.divider()


def _display_pdf_summary(file_details: dict):
    """Display the PDF summary section."""
    st.markdown("**Résumé du PDF (généré par IA):**")
    st.markdown(file_details.get('summary', "*Aucun résumé disponible.*"))


def _display_user_notes_section(selected_file_id: str, file_details: dict):
    """Display the user notes text area."""
    current_notes = file_details.get('user_notes', '')
    st.text_area(
        "Notes additionnelles sur ce fichier (utilisées par l'agent RAG)",
        value=current_notes,
        key=f"user_notes_{selected_file_id}",
        placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur ce fichier...",
        height=150,
        on_change=update_user_notes_callback,
        args=(selected_file_id,)
    )


def _display_classification_section(selected_file_id: str, file_details: dict):
    """Display the classification selection section."""
    current_classification = file_details.get('classification')
    suggested_classification = file_details.get('suggested_classification')
    pre_selected_value = current_classification or suggested_classification
    
    try:
        default_index = PDF_CLASSES.index(pre_selected_value) if pre_selected_value in PDF_CLASSES else 0
    except ValueError:
        default_index = 0

    selected_class = st.selectbox(
        "Confirmez ou modifiez la classification de ce document :",
        PDF_CLASSES,
        key=f'pdf_classification_{selected_file_id}',
        index=default_index,
        on_change=update_classification_callback,
        args=(selected_file_id,)
    )

    if selected_class != PDF_CLASSES[0] and st.session_state.processed_files[selected_file_id].get('classification') is None:
        st.session_state.processed_files[selected_file_id]['classification'] = selected_class
        st.session_state.processed_files[selected_file_id]['status'] = 'classified'
        save_persistent_state(st.session_state.processed_files)


def _display_indexing_section(selected_file_id: str, file_details: dict, mistral_api_key: str):
    """Display the indexing buttons section."""
    classification_for_indexing = st.session_state.processed_files[selected_file_id].get('classification')
    is_indexed = st.session_state.processed_files[selected_file_id].get('indexed', False)

    disable_button = not classification_for_indexing or is_indexed
    button_text = "Indexer le PDF" if not is_indexed else "PDF Déjà Indexé"

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(button_text, key=f"index_pdf_{selected_file_id}", disabled=disable_button, use_container_width=True):
             if classification_for_indexing and not is_indexed:
                 filename = file_details.get('filename', 'Unknown')
                 
                 # Add simple start notification to chat
                 start_message = f"🔄 Indexation de **{filename}** démarrée..."
                 st.session_state.messages.append({
                     "role": "assistant", 
                     "content": start_message,
                     "timestamp": time.time()
                 })
                 
                 # Update status immediately
                 sync_file_status(selected_file_id, 'indexing', None, force_rerun=False)
                 
                 # Create a progress placeholder for real-time updates
                 progress_placeholder = st.empty()
                 
                 with progress_placeholder.container():
                     st.info("🔄 Indexation en cours, veuillez patienter...")
                     
                     # Perform the actual indexing
                     updated_details = index_pdf(
                         file_id=selected_file_id,
                         file_details=file_details,
                         session_id=st.session_state.session_id,
                         mistral_api_key=mistral_api_key
                     )
                 
                 # Clear the progress placeholder
                 progress_placeholder.empty()
                 
                 if updated_details:
                     st.session_state.processed_files[selected_file_id] = updated_details
                     save_persistent_state(st.session_state.processed_files)
                     
                     final_status = updated_details.get('status', 'error_indexing')
                     
                     if final_status == 'indexed':
                         # Add simple success notification to chat
                         success_message = f"✅ **{filename}** indexé avec succès ! Vous pouvez poser vos questions."
                         st.session_state.messages.append({
                             "role": "assistant", 
                             "content": success_message,
                             "timestamp": time.time()
                         })
                         
                         sync_file_status(selected_file_id, 'indexed', None, force_rerun=False)
                         st.session_state.selected_file_id_for_action = None
                         st.success(f"✅ '{filename}' indexé avec succès !")
                     else:
                         # Add simple error notification to chat
                         error_message = f"❌ Échec de l'indexation de **{filename}**."
                         st.session_state.messages.append({
                             "role": "assistant", 
                             "content": error_message,
                             "timestamp": time.time()
                         })
                         
                         sync_file_status(selected_file_id, final_status, None, force_rerun=False)
                         st.error(f"❌ Erreur lors de l'indexation de '{filename}'")
                     
                     # Force final UI update
                     st.rerun()
                 else:
                     # Add simple failure notification to chat
                     failure_message = f"❌ Échec de l'indexation de **{filename}**."
                     st.session_state.messages.append({
                         "role": "assistant", 
                         "content": failure_message,
                         "timestamp": time.time()
                     })
                     
                     sync_file_status(selected_file_id, 'error_indexing', None, force_rerun=False)
                     st.error(f"❌ Erreur lors de l'indexation de '{filename}'")
                     st.rerun() 