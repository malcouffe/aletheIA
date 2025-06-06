"""
File management functionality including cleanup and deletion.
"""
import streamlit as st
import os
import shutil
from core.session_manager import save_persistent_state


def cleanup_single_file_resources(file_id: str, details: dict):
    """Attempts to clean up resources associated with a single file (temp files, DBs, figures)."""
    temp_path = details.get('temp_path')
    if temp_path and os.path.exists(temp_path):
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
                print(f"Removed temp file: {temp_path}")
                temp_dir = os.path.dirname(temp_path)
                if os.path.basename(temp_dir) == file_id:
                     try:
                         os.rmdir(temp_dir)
                         print(f"Removed temp directory: {temp_dir}")
                         session_dir = os.path.dirname(temp_dir)
                         if not os.listdir(session_dir) and session_dir != os.path.join("data", "temp_files"):
                             os.rmdir(session_dir)
                             print(f"Removed session temp directory: {session_dir}")
                     except OSError as e:
                         st.warning(f"Could not remove directory {temp_dir} (maybe not empty or permissions?): {e}")
            elif os.path.isdir(temp_path): 
                shutil.rmtree(temp_path)
                print(f"Removed temp directory (shutil): {temp_path}")
        except Exception as e:
            st.warning(f"Error removing temp resource {temp_path}: {e}")

    db_path = details.get('db_path')
    if db_path and os.path.exists(db_path):
        try:
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
                print(f"Removed vector DB directory: {db_path}")
                class_dir = os.path.dirname(db_path)
                try:
                    if not os.listdir(class_dir):
                        os.rmdir(class_dir)
                        print(f"Removed classification DB directory: {class_dir}")
                        session_db_dir = os.path.dirname(class_dir)
                        if not os.listdir(session_db_dir) and session_db_dir != os.path.join("data", "vector_db"):
                            os.rmdir(session_db_dir)
                            print(f"Removed session DB directory: {session_db_dir}")
                except OSError as e:
                    st.warning(f"Could not remove DB directory {class_dir} (maybe not empty or permissions?): {e}")
            else:
                 st.warning(f"Expected DB path {db_path} to be a directory, but it is not. Cleanup skipped.")
        except Exception as e:
            st.warning(f"Error removing vector DB {db_path}: {e}")

    try:
        session_id = st.session_state.session_id
        figures_dir = os.path.join('data', 'figures', session_id, file_id)
        if os.path.exists(figures_dir) and os.path.isdir(figures_dir):
            shutil.rmtree(figures_dir)
            print(f"Removed figures directory: {figures_dir}")
            session_figures_dir = os.path.dirname(figures_dir)
            try:
                if not os.listdir(session_figures_dir) and session_figures_dir != os.path.join('data', 'figures'):
                    os.rmdir(session_figures_dir)
                    print(f"Removed session figures directory: {session_figures_dir}")
            except OSError as e:
                 st.warning(f"Could not remove session figures directory {session_figures_dir} (maybe not empty or permissions?): {e}")
    except AttributeError:
         st.warning(f"Could not clean up figures for {file_id}: session_id not found in st.session_state.")
    except Exception as e:
        st.warning(f"Error removing figures directory for {file_id}: {e}")


def delete_file_callback(file_id_to_delete: str):
    """Callback function to handle file deletion."""
    if 'processed_files' in st.session_state and file_id_to_delete in st.session_state.processed_files:
        details = st.session_state.processed_files[file_id_to_delete]
        filename_for_message = details.get('filename', file_id_to_delete)
        cleanup_single_file_resources(file_id_to_delete, details)
        del st.session_state.processed_files[file_id_to_delete]
        save_persistent_state(st.session_state.processed_files)
        st.success(f"Fichier '{filename_for_message}' et ses ressources associées ont été supprimés.")
        if st.session_state.get('selected_file_id_for_action') == file_id_to_delete:
            st.session_state.selected_file_id_for_action = None
        st.rerun()
    else:
        st.error(f"Impossible de supprimer le fichier : ID '{file_id_to_delete}' non trouvé.")


def select_pdf_for_action(file_id: str):
    """Sets the file_id for the PDF selected for classification/indexing."""
    st.session_state.selected_file_id_for_action = file_id 