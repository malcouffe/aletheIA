"""
Sidebar UI components for displaying processed files and their status.
"""
import streamlit as st
from config.settings import STATUS_MAP, FILE_TYPE_ICONS
from ui.state_sync import get_status_emoji
from core.session_manager import update_user_notes_callback, update_csv_user_notes_callback
from core.file_manager import delete_file_callback, select_pdf_for_action


def display_processed_files_sidebar():
    """Displays the list of processed files and their status in the sidebar."""
    with st.sidebar:
        st.divider()
        
        # Simplified header - just the title, auto-sync in background
        st.subheader("Fichiers Traités")

        processed_files = st.session_state.get('processed_files', {})

        if not processed_files:
            st.info("Aucun fichier traité dans cette session.")
            return

        sorted_file_ids = sorted(
            processed_files.keys(),
            key=lambda fid: (
                processed_files[fid].get('filename', '').lower(), 
                fid
            )
        )

        for file_id in sorted_file_ids:
            details = processed_files[file_id]
            file_type = details.get('type', 'unknown')
            filename = details.get('filename', f"ID: {file_id}")
            status = details.get('status', 'unknown')

            icon = FILE_TYPE_ICONS.get(file_type, FILE_TYPE_ICONS['unknown'])
            
            # Use our centralized status emoji function
            status_indicator = get_status_emoji(status)
            title = f"{icon} {filename} {status_indicator}"
            
            # Highlight indexing files
            expanded = status == 'indexing'
            
            with st.expander(title, expanded=expanded):
                _display_file_status(status)
                _display_file_actions(file_id, file_type, status)
                _display_file_details(file_id, file_type, details)
                
                # Only keep the delete button - sync happens automatically
                with st.columns([3,2])[1]:
                    st.button(
                        "Supprimer", 
                        key=f"delete_{file_id}", 
                        on_click=delete_file_callback, 
                        args=(file_id,), 
                        type="secondary", 
                        use_container_width=True
                    )
            st.markdown("---")
        
        # Auto-sync happens transparently when needed
        _auto_sync_if_needed()


def _display_file_status(status: str):
    """Display the status of a file with appropriate styling."""
    status_display_map = {
        'awaiting_classification': (st.warning, "⏳ En attente de classification"),
        'classified': (st.info, "📝 Classifié, prêt pour indexation"),
        'indexing': (st.info, "🔄 Indexation en cours..."),
        'indexed': (st.success, "✅ Indexé et prêt pour RAG"),
        'ready': (st.success, "✅ Prêt pour analyse (CSV)"),
        'error_extraction': (st.error, "❌ Erreur d'extraction de texte"),
        'error_analysis': (st.error, "❌ Erreur d'analyse IA"),
        'error_validation': (st.error, "❌ Erreur de validation CSV"),
        'error_indexing': (st.error, "❌ Erreur d'indexation"),
        'error_indexing_missing_temp': (st.error, "❌ Erreur Indexation (Fichier temp. manquant)"),
        'unknown': (st.error, "❓ Statut inconnu")
    }
    
    display_func, status_text = status_display_map.get(status, (st.error, f"❓ Statut non géré: {status}"))
    
    # Special handling for indexing status with animated indicator
    if status == 'indexing':
        with st.container():
            st.info(status_text)
            # Add a progress indicator for indexing
            progress_bar = st.progress(0)
            import time
            for i in range(3):
                progress_bar.progress((i + 1) * 33)
                time.sleep(0.1)
            progress_bar.progress(100)
    else:
        display_func(f"Statut: {status_text}")


def _display_file_actions(file_id: str, file_type: str, status: str):
    """Display action buttons for files."""
    action_col, delete_col = st.columns([3,2])

    with action_col:
        if file_type == 'pdf' and status in ['awaiting_classification', 'classified', 'error_indexing', 'indexed']:
            button_label = "Classifier/Indexer" if status != 'indexed' else "Re-classifier/Indexer"
            if st.button(
                button_label, 
                key=f"select_pdf_action_{file_id}", 
                on_click=select_pdf_for_action, 
                args=(file_id,), 
                use_container_width=True
            ):
                st.rerun()


def _display_file_details(file_id: str, file_type: str, details: dict):
    """Display detailed information about a file."""
    if file_type == 'pdf':
        _display_pdf_details(file_id, details)
    elif file_type == 'csv':
        _display_csv_details(file_id, details)


def _display_pdf_details(file_id: str, details: dict):
    """Display PDF-specific details."""
    classification = details.get('classification')
    if classification:
        st.write(f"Classification: {classification}")
    
    summary = details.get('summary')
    if summary:
        show_summary = st.checkbox("Afficher le résumé", key=f"show_summary_{file_id}")
        if show_summary:
            st.markdown(f"**Résumé:** {summary}")


def _display_csv_details(file_id: str, details: dict):
    """Display CSV-specific details."""
    csv_args = details.get('csv_args', {})
    separator = csv_args.get('separator')
    rows = details.get('rows', 'N/A')
    columns = details.get('columns', [])
    cols = len(columns)
    size_mb = details.get('size_mb', 0)
    
    st.write(f"📊 **{rows}** lignes, **{cols}** colonnes")
    if separator:
        st.write(f"Séparateur: `{separator}`")
    if size_mb:
        st.write(f"Taille: {size_mb:.1f} MB")
    
    # Show first few column names
    if columns:
        col_preview = ', '.join(columns[:3])
        if len(columns) > 3:
            col_preview += f"... (+{len(columns)-3} autres)"
        st.write(f"Colonnes: {col_preview}")
    
    # User notes for CSV
    current_csv_notes = details.get('user_notes', '')
    st.text_area(
        "Notes contextuelles (utilisées par l'agent d'analyse)",
        value=current_csv_notes,
        key=f"csv_user_notes_{file_id}",
        placeholder="Ajoutez ici du contexte sur les données, objectifs d'analyse, ou informations spécifiques...",
        height=100,
        on_change=update_csv_user_notes_callback,
        args=(file_id,)
    ) 

def _auto_sync_if_needed():
    """
    Automatically sync UI state only when necessary (when there are indexing files).
    This replaces the old manual sync buttons with intelligent background sync.
    """
    processed_files = st.session_state.get('processed_files', {})
    
    # Check if any files are in active states that need monitoring
    active_states = ['indexing']
    has_active_files = any(
        details.get('status') in active_states 
        for details in processed_files.values()
    )
    
    # Only auto-refresh if there are files in active states
    if has_active_files:
        # Add a small delay and refresh (less aggressive than before)
        import time
        time.sleep(1)  # Reduced from 2 seconds to 1 second
        st.rerun() 