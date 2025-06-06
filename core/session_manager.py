"""
Session state management and persistence functionality.
"""
import streamlit as st
import os
import json
import shutil
from config.settings import PERSISTENCE_DIR, PERSISTENCE_FILE


def save_persistent_state(processed_files_dict):
    """Saves the processed_files dictionary to a JSON file."""
    try:
        os.makedirs(PERSISTENCE_DIR, exist_ok=True)
        with open(PERSISTENCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_files_dict, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save session state: {e}")


def load_persistent_state():
    """Loads and validates the processed_files dictionary from a JSON file."""
    if not os.path.exists(PERSISTENCE_FILE):
        return {}

    try:
        with open(PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        st.error("Failed to load session state: Invalid JSON format. Starting fresh.")
        try:
            corrupted_backup_path = PERSISTENCE_FILE + ".corrupted"
            shutil.copy(PERSISTENCE_FILE, corrupted_backup_path)
            st.warning(f"Corrupted state file backed up to {corrupted_backup_path}")
        except Exception as backup_e:
             st.warning(f"Could not back up corrupted state file: {backup_e}")
        return {}
    except Exception as e:
        st.error(f"Failed to load session state: {e}. Starting fresh.")
        return {}

    validated_data = {}
    for file_id, details in loaded_data.items():
        is_valid = True
        file_type = details.get('type')
        filename = details.get('filename', f"ID: {file_id}") 

        if file_type == 'pdf':
            if details.get('indexed') and not os.path.exists(str(details.get('db_path'))):
                is_valid = False
                st.warning(f"Validation: PDF '{filename}' marked as indexed but db_path '{details.get('db_path')}' not found. Invalidating.")
            elif details.get('status') == 'awaiting_classification' and 'temp_path' in details and not os.path.exists(str(details.get('temp_path'))):
                 is_valid = False
                 st.warning(f"Validation: PDF '{filename}' awaiting classification but temp_path '{details.get('temp_path')}' not found. Invalidating.")

        elif file_type == 'csv':
            csv_args = details.get('csv_args', {})
            source_file = csv_args.get('source_file')
            if details.get('status') == 'ready' and (not source_file or not os.path.exists(str(source_file))):
                is_valid = False
                st.warning(f"Validation: CSV '{filename}' marked as ready but source_file '{source_file}' not found. Invalidating.")
            figures_dir = csv_args.get('figures_dir')
            if figures_dir and not os.path.exists(str(figures_dir)):
                 print(f"Validation: Figures directory '{figures_dir}' for CSV '{filename}' not found.")

        if is_valid:
            validated_data[file_id] = details
        else:
            st.warning(f"File entry for '{filename}' (ID: {file_id}) was invalid and will be cleaned up if possible.")
            from core.file_manager import cleanup_single_file_resources
            cleanup_single_file_resources(file_id, details)

    return validated_data


def update_user_notes_callback(file_id: str):
    """Updates the user notes for the specified PDF file in session state."""
    notes_key = f"user_notes_{file_id}"
    if notes_key in st.session_state:
        st.session_state.processed_files[file_id]['user_notes'] = st.session_state[notes_key]
        save_persistent_state(st.session_state.processed_files)


def update_csv_user_notes_callback(file_id: str):
    """Updates the user notes for the specified CSV file in session state."""
    notes_key = f"csv_user_notes_{file_id}"
    if notes_key in st.session_state:
        st.session_state.processed_files[file_id]['user_notes'] = st.session_state[notes_key]
        save_persistent_state(st.session_state.processed_files)


def update_classification_callback(file_id: str):
    """Updates the classification for the selected PDF file in session state."""
    from ui.components import PDF_CLASSES
    
    new_classification = st.session_state[f"pdf_classification_{file_id}"]
    current_details = st.session_state.processed_files[file_id]
    
    new_classification_value = None
    new_status = current_details.get('status', 'awaiting_classification')

    if new_classification == PDF_CLASSES[0]:
        new_classification_value = None
        if current_details.get('classification') or current_details.get('indexed'):
            new_status = 'awaiting_classification'
    else:
        new_classification_value = new_classification
        new_status = 'classified'
        
    classification_changed = (current_details.get('classification') != new_classification_value)
    
    st.session_state.processed_files[file_id]['classification'] = new_classification_value
    st.session_state.processed_files[file_id]['status'] = new_status

    if classification_changed:
        st.session_state.processed_files[file_id]['indexed'] = False
        st.session_state.processed_files[file_id]['db_path'] = None 
        print(f"Classification changed for {file_id}. Marked as not indexed.")

    save_persistent_state(st.session_state.processed_files) 