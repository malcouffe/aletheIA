import streamlit as st
import os
import shutil
import uuid
import torch
import json
from smolagents import OpenAIServerModel
from ui_components import handle_uploaded_file, index_pdf
import ui_components

torch.classes.__path__ = []

# --- Persistence Constants ---
PERSISTENCE_DIR = os.path.join("data", "session_persistence")
PERSISTENCE_FILE = os.path.join(PERSISTENCE_DIR, "persistent_session_state.json")

# --- Persistence Functions ---
def save_persistent_state(processed_files_dict):
    """Saves the processed_files dictionary to a JSON file."""
    try:
        os.makedirs(PERSISTENCE_DIR, exist_ok=True)
        with open(PERSISTENCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_files_dict, f, indent=4)
        print(f"Session state successfully saved to {PERSISTENCE_FILE}") # Optional: for debugging
    except Exception as e:
        st.error(f"Failed to save session state: {e}")

def load_persistent_state():
    """Loads and validates the processed_files dictionary from a JSON file."""
    if not os.path.exists(PERSISTENCE_FILE):
        print("Persistence file not found. Starting with empty state.") # Optional: for debugging
        return {}

    try:
        with open(PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        st.error("Failed to load session state: Invalid JSON format. Starting fresh.")
        # Optionally back up the corrupted file
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
        # --- Validation Logic ---
        is_valid = True
        file_type = details.get('type')
        filename = details.get('filename', f"ID: {file_id}") # For logging

        if file_type == 'pdf':
            # If indexed, db_path MUST exist
            if details.get('indexed') and not os.path.exists(str(details.get('db_path'))):
                print(f"Validation failed for {filename}: Indexed PDF's db_path '{details.get('db_path')}' not found. Discarding.")
                is_valid = False
            # If not indexed, temp_path might exist (though usually cleaned up after indexing)
            # Allow entries awaiting classification even if temp path is missing (might be error state)
            # Allow indexed entries even if temp_path is missing (expected)
            # But if it's supposed to have a temp path and doesn't, that's an issue.
            elif details.get('status') == 'awaiting_classification' and 'temp_path' in details and not os.path.exists(str(details.get('temp_path'))):
                 print(f"Validation failed for {filename}: PDF awaiting classification's temp_path '{details.get('temp_path')}' not found. Discarding.")
                 # Let's keep error states even if path is missing
                 # is_valid = False # Re-evaluate if we should keep error states

        elif file_type == 'csv':
            # CSV needs its source file if it's in 'ready' state
            csv_args = details.get('csv_args', {})
            source_file = csv_args.get('source_file')
            if details.get('status') == 'ready' and (not source_file or not os.path.exists(str(source_file))):
                print(f"Validation failed for {filename}: CSV's source_file '{source_file}' not found. Discarding.")
                is_valid = False
            # Also check figures dir existence? Maybe less critical.
            figures_dir = csv_args.get('figures_dir')
            if figures_dir and not os.path.exists(str(figures_dir)):
                 print(f"Note for {filename}: CSV's figures_dir '{figures_dir}' not found. May need regeneration.")
                 # Don't invalidate just for figures dir

        # Add other file types' validation if needed

        if is_valid:
            validated_data[file_id] = details
        else:
            # If invalid, attempt cleanup of associated resources like dangling DB/figure dirs
            # Call the existing cleanup function, it handles non-existent paths gracefully
             _cleanup_single_file_resources(file_id, details)


    print(f"Loaded {len(validated_data)} valid entries from state file.") # Optional: debugging
    return validated_data

# --- Callback to update user notes ---
def _update_user_notes_callback(file_id: str):
    """Updates the user notes for the specified PDF file in session state."""
    notes_key = f"user_notes_{file_id}"
    if notes_key in st.session_state:
        st.session_state.processed_files[file_id]['user_notes'] = st.session_state[notes_key]
        # Save state after updating notes
        save_persistent_state(st.session_state.processed_files)

def display_pdf_action_section(mistral_api_key: str):
    """Displays the UI section for PDF classification and indexing for a selected PDF."""
    selected_file_id = st.session_state.get('selected_file_id_for_action')

    if not selected_file_id or selected_file_id not in st.session_state.get('processed_files', {}):
        # Nothing selected or selection is invalid, display nothing or an info message
        return

    file_details = st.session_state.processed_files[selected_file_id]

    # Ensure it's a PDF
    if file_details.get('type') != 'pdf':
        st.warning("Le fichier sélectionné n'est pas un PDF.")
        return

    st.subheader(f"Classification/Indexation : {file_details.get('filename', selected_file_id)}")
    st.markdown("**Résumé du PDF (généré par IA):**")
    st.markdown(file_details.get('summary', "*Aucun résumé disponible.*"))

    # --- Moved User Notes Text Area ---
    current_notes = file_details.get('user_notes', '')
    st.text_area(
        "Notes additionnelles sur ce fichier",
        value=current_notes,
        key=f"user_notes_{selected_file_id}", # Unique key
        placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur ce fichier...",
        height=150,
        on_change=_update_user_notes_callback,
        args=(selected_file_id,)
    )
    # -----------------------------------

    current_classification = file_details.get('classification')

    # Determine default index for selectbox
    suggested_classification = file_details.get('suggested_classification')
    pre_selected_value = current_classification or suggested_classification
    try:
        default_index = ui_components.PDF_CLASSES.index(pre_selected_value) if pre_selected_value in ui_components.PDF_CLASSES else 0
    except ValueError:
        default_index = 0

    # Classification Selectbox (uses file_id specific key and callback)
    selected_class = st.selectbox(
        "Confirmez ou modifiez la classification de ce document :",
        ui_components.PDF_CLASSES,
        key=f'pdf_classification_{selected_file_id}', # Unique key
        index=default_index,
        on_change=_update_classification_callback, # Use the callback
        args=(selected_file_id,) # Pass file_id to callback
    )

    # --- Ensure state reflects the current valid selection in the box ---
    if selected_class != ui_components.PDF_CLASSES[0] and st.session_state.processed_files[selected_file_id].get('classification') is None:
        st.session_state.processed_files[selected_file_id]['classification'] = selected_class
        st.session_state.processed_files[selected_file_id]['status'] = 'classified'
    # --------------------------------------------------------------------

    # Get the *current* classification from state for button logic
    classification_for_indexing = st.session_state.processed_files[selected_file_id].get('classification')
    is_indexed = st.session_state.processed_files[selected_file_id].get('indexed', False)

    # Indexing Button
    disable_button = not classification_for_indexing or is_indexed
    button_text = "Indexer le PDF" if not is_indexed else "PDF Déjà Indexé"

    col1, col2 = st.columns(2)
    with col1:
        if st.button(button_text, key=f"index_pdf_{selected_file_id}", disabled=disable_button, use_container_width=True):
             if classification_for_indexing and not is_indexed:
                 # --- Indexing ---
                 updated_details = index_pdf(
                     file_id=selected_file_id,
                     file_details=file_details,
                     session_id=st.session_state.session_id,
                     mistral_api_key=mistral_api_key
                 )

                 if updated_details:
                     st.session_state.processed_files[selected_file_id] = updated_details
                     # Save state after successful indexing
                     save_persistent_state(st.session_state.processed_files)
                     if updated_details.get('status') == 'indexed':
                         st.session_state.selected_file_id_for_action = None
                     st.rerun()
                 else:
                     st.error(f"L'indexation a échoué de manière inattendue pour {selected_file_id}.")
                     st.rerun()
             else:
                  st.warning(f"Impossible d'indexer : Classification manquante ou déjà indexé pour {selected_file_id}.")
    
    # Add a button to close/deselect this section
    with col2:
        if st.button("Terminer", key=f"close_indexing_{selected_file_id}", type="secondary", use_container_width=True):
            st.session_state.selected_file_id_for_action = None
            st.rerun()
            
    st.divider()

# --- Functions for Multi-File Handling ---

def _cleanup_single_file_resources(file_id: str, details: dict):
    """Attempts to clean up resources associated with a single file (temp files, DBs, figures)."""
    temp_path = details.get('temp_path')

    if temp_path and os.path.exists(temp_path):
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
                temp_dir = os.path.dirname(temp_path)
                if os.path.basename(temp_dir) == file_id:
                     try:
                         os.rmdir(temp_dir)
                         session_dir = os.path.dirname(temp_dir)
                         if not os.listdir(session_dir):
                             os.rmdir(session_dir)
                     except OSError as e:
                         st.warning(f"Could not remove directory {temp_dir} (maybe not empty or permissions?): {e}")
            elif os.path.isdir(temp_path): # Should not happen based on current logic, but check
                shutil.rmtree(temp_path)
        except Exception as e:
            st.warning(f"Error removing temp resource {temp_path}: {e}")

    # Cleanup vector database
    db_path = details.get('db_path')
    if db_path and os.path.exists(db_path):
        try:
            # DB path is expected to be a directory containing index files
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
                class_dir = os.path.dirname(db_path)
                try:
                    if not os.listdir(class_dir):
                        os.rmdir(class_dir)
                except OSError as e:
                    st.warning(f"Could not remove directory {class_dir} (maybe not empty or permissions?): {e}")
            else:
                 st.warning(f"Expected DB path {db_path} to be a directory, but it is not.")
        except Exception as e:
            st.warning(f"Error removing vector DB {db_path}: {e}")

    # --- Cleanup Figures Directory --- 
    try:
        session_id = st.session_state.session_id # Assuming session_id is accessible
        figures_dir = os.path.join('data', 'figures', session_id, file_id)
        if os.path.exists(figures_dir) and os.path.isdir(figures_dir):
            shutil.rmtree(figures_dir)
            print(f"Removed figures directory: {figures_dir}")
            # Optionally, try removing session dir if empty (like temp dir logic)
            session_figures_dir = os.path.dirname(figures_dir)
            try:
                if not os.listdir(session_figures_dir):
                    os.rmdir(session_figures_dir)
            except OSError as e:
                 st.warning(f"Could not remove session figures directory {session_figures_dir} (maybe not empty or permissions?): {e}")
    except AttributeError:
         st.warning(f"Could not clean up figures for {file_id}: session_id not found in st.session_state.")
    except Exception as e:
        st.warning(f"Error removing figures directory for {file_id}: {e}")
    # ---------------------------------

def _delete_file_callback(file_id_to_delete: str):
    """Callback function to handle file deletion."""
    if 'processed_files' in st.session_state and file_id_to_delete in st.session_state.processed_files:
        details = st.session_state.processed_files[file_id_to_delete]
        _cleanup_single_file_resources(file_id_to_delete, details)
        del st.session_state.processed_files[file_id_to_delete]
        # Save state after deletion
        save_persistent_state(st.session_state.processed_files)
        st.success(f"Fichier '{details.get('filename', file_id_to_delete)}' supprimé.")
        st.rerun()
    else:
        st.error(f"Impossible de supprimer le fichier : ID '{file_id_to_delete}' non trouvé.")

# Renamed function and updated logic
def display_processed_files_sidebar():
    """Displays the list of processed files and their status in the sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("Fichiers Traités")

        processed_files = st.session_state.get('processed_files', {})

        if not processed_files:
            st.info("Aucun fichier traité dans cette session.")
            return

        sorted_file_ids = sorted(processed_files.keys(), key=lambda fid: processed_files[fid].get('filename', ''))

        for file_id in sorted_file_ids:
            details = processed_files[file_id]
            file_type = details.get('type', 'unknown')
            filename = details.get('filename', f"ID: {file_id}")
            status = details.get('status', 'unknown')

            icon = "📄" if file_type == 'pdf' else ("📊" if file_type == 'csv' else "❓")
            st.markdown(f"**{icon} {filename}**")

            # Display status with appropriate message type
            status_map = {
                'awaiting_classification': (st.warning, "En attente de classification"),
                'classified': (st.info, "Classifié, prêt pour indexation"),
                'indexing': (st.spinner, "Indexation en cours..."), # TODO: Need a real 'indexing' status?
                'indexed': (st.success, "Indexé"),
                'ready': (st.success, "Prêt pour analyse"),
                'error_extraction': (st.error, "Erreur d'extraction de texte"),
                'error_analysis': (st.error, "Erreur d'analyse IA"),
                'error_validation': (st.error, "Erreur de validation CSV"),
                'error_indexing': (st.error, "Erreur d'indexation"),
                'error_indexing_missing_temp': (st.error, "Erreur Indexation (Fichier temporaire manquant)"),
                'unknown': (st.error, "Statut inconnu")
            }
            display_func, status_text = status_map.get(status, (st.error, f"Statut non géré: {status}"))
            col1, col2 = st.columns([3, 1])
            with col1:
                display_func(f"Status: {status_text}")
            with col2:
                if file_type == 'pdf' and status in ['awaiting_classification', 'classified', 'error_indexing']:
                    st.button("Indexer", key=f"select_{file_id}", on_click=_select_pdf_for_action, args=(file_id,))
            
            # Display specific details
            if file_type == 'pdf':
                classification = details.get('classification')
                if classification:
                    st.write(f"Classification: {classification}")
                summary = details.get('summary')
                if summary:
                    with st.expander("Afficher/Masquer le résumé"):
                         st.markdown(summary)

            elif file_type == 'csv':
                csv_args = details.get('csv_args', {})
                separator = csv_args.get('separator')
                rows = details.get('rows', 'N/A')
                cols = len(details.get('columns', []))
                if separator:
                    st.write(f"Séparateur: '{separator}'")
                st.write(f"Lignes: {rows}, Colonnes: {cols}")

            # Delete button for each file (use columns for layout)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                 st.button("Supprimer", key=f"delete_{file_id}", on_click=_delete_file_callback, args=(file_id,), type="secondary", use_container_width=True)

            st.divider()

# --- Callbacks for Multi-File UI --- 

def _select_pdf_for_action(file_id: str):
    """Sets the file_id for the PDF selected for classification/indexing."""
    st.session_state.selected_file_id_for_action = file_id

def _update_classification_callback(file_id: str):
    """Updates the classification for the selected PDF file in session state."""
    new_classification = st.session_state[f"pdf_classification_{file_id}"]
    current_details = st.session_state.processed_files[file_id]
    
    if new_classification == ui_components.PDF_CLASSES[0]: # "Select Classification..."
        new_classification_value = None
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

    # Add save state call here
    save_persistent_state(st.session_state.processed_files)

def build_manager_prompt(user_query, csv_args, pdf_context):
    """Builds the prompt for the manager agent based on context."""
    # TODO: This needs significant update for multi-file context
    return f"""
    Your goal is to route the user query to the most appropriate specialized agent.
    User Query: \"{user_query}\"

    Available agents (provided in additional_args):
    - search_agent: For general web searches.
    - data_analyst: For analyzing the provided CSV data.
    - rag_agent: For answering questions based on the indexed PDF document.

    Context:
    - CSV Loaded: {'Yes' if csv_args else 'No'}
    - PDF Indexed: {'Yes' if pdf_context else 'No'}

    Decision Logic:
    1. If a PDF is indexed ('PDF Indexed: Yes') and the user query is relevant, use rag_agent.
    2. If a CSV is loaded ('CSV Loaded: Yes'), use data_analyst.
    3. Otherwise, use search_agent.

    Instructions:
    - Analyze the query and context.
    - Generate Python code to call the .run() method of the chosen agent.
    - Use the provided agent instances (search_agent, data_analyst, rag_agent) from additional_args.
    - Pass the necessary arguments (e.g., query, csv_args, pdf_context) to the chosen agent's .run() method.
    - Example for RAG: `result = rag_agent.run(query=\"{user_query}\", pdf_context=pdf_context)`
    - Example for Data Analyst: `result = data_analyst.run(query=\"{user_query}\", additional_args={{'csv_analyzer': csv_args}})`
    - Example for Search: `result = search_agent.run(\"{user_query}\")`
    """

def main():

    st.title("Assistant d'Analyse de Documents (PDF/CSV)")
    st.write("Chargez un fichier PDF ou CSV, ou configurez vos clés API dans la barre latérale pour commencer.")

    # --- Initialize session state ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # Initialize processed_files: Load from persistent storage OR start empty
        if 'processed_files' not in st.session_state:
             st.session_state.processed_files = load_persistent_state()

    # Initialize chat history (remains session-specific)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Configuration de la Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        api_key_from_env = os.environ.get("OPENAI_API_KEY")
        mistral_api_key_from_env = os.environ.get("MISTRAL_API_KEY")

        if not api_key_from_env:
            api_key = st.text_input("Clé API OpenAI", type="password")
        else:
            api_key = api_key_from_env # Assign the key regardless
            # Show toast only once per session if the key is found
            if not st.session_state.get('openai_key_toast_shown', False):
                st.toast("Clé API OpenAI trouvée", icon="✅")
                st.session_state.openai_key_toast_shown = True # Set the flag
            
        if not mistral_api_key_from_env:
             mistral_api_key = st.text_input("Clé API Mistral (pour PDF)", type="password")
        else:
            mistral_api_key = mistral_api_key_from_env # Assign the key regardless
            # Show toast only once per session if the key is found
            if not st.session_state.get('mistral_key_toast_shown', False):
                st.toast("Clé API Mistral trouvée", icon="✅")
                st.session_state.mistral_key_toast_shown = True # Set the flag

        st.subheader("Chargement de Fichier")
        uploaded_file = st.file_uploader(
            "Charger un fichier (PDF, CSV, TXT)",
            type=['pdf', 'csv', 'txt'],
            key="unified_uploader"
        )


        # CSV specific option
        use_memory_limit = True

    # -------------------------------------

    # --- Interface Principale ---
    model = None
    if api_key:
        try:
            model = OpenAIServerModel(
                model_id="gpt-4o",
                api_base="https://api.openai.com/v1",
                api_key=api_key,
            )
        except Exception as e:
            st.warning(f"Impossible d'initialiser le modèle OpenAI : {e}")

    # --- File Upload Handling (using functions from ui_components) ---
    # Call the unified file handler if a file was uploaded
    if uploaded_file is not None:
        processed_file_info = handle_uploaded_file(
            uploaded_file=uploaded_file,
            session_id=st.session_state.session_id,
            model=model,
            mistral_api_key=mistral_api_key,
            use_memory_limit=use_memory_limit,
        )

        if processed_file_info and 'file_id' in processed_file_info and 'details' in processed_file_info:
            file_id = processed_file_info['file_id']
            details = processed_file_info['details']

            # --- Clear chat history and add upload confirmation ---
            st.session_state.messages = [] # Clear previous chat
            st.session_state.processed_files[file_id] = details
            # Save state after adding a new file
            save_persistent_state(st.session_state.processed_files)

            filename = details.get('filename', 'Inconnu')
            file_type = details.get('type', 'unknown')
            status = details.get('status', 'unknown')
            # Use status mapping for user-friendly text (borrowed from sidebar logic)
            status_map = {
                'awaiting_classification': "en attente de classification",
                'classified': "classifié, prêt pour indexation",
                'indexing': "indexation en cours...",
                'indexed': "indexé",
                'ready': "prêt pour analyse",
                'error_extraction': "erreur d'extraction de texte",
                'error_analysis': "erreur d'analyse IA",
                'error_validation': "erreur de validation CSV",
                'error_indexing': "erreur d'indexation",
                'error_indexing_missing_temp': "Erreur Indexation (Fichier temporaire manquant)",
                'unknown': "statut inconnu"
            }
            status_text = status_map.get(status, f"statut non géré: {status}")

            upload_message = f"Fichier '{filename}' ({file_type.upper()}) téléversé et traité. Statut: {status_text}."
            if file_type == 'pdf' and status == 'awaiting_classification':
                upload_message += " Cliquez sur 'Indexer' dans la barre latérale pour continuer."
            elif file_type == 'csv' and status == 'ready':
                 upload_message += " Vous pouvez maintenant poser des questions sur ses données."

            st.session_state.messages.append({"role": "assistant", "content": upload_message})

            st.success(f"Fichier '{filename}' traité et ajouté à la session. Le chat a été réinitialisé.")
            st.rerun() # Rerun needed here to show cleared chat + new message

        elif processed_file_info:
            st.warning("Le traitement du fichier a terminé mais n'a pas pu être ajouté à l'état de la session.")

    # --- PDF Classification/Indexing UI (Conditional) ---
    display_pdf_action_section(mistral_api_key)

    # --- Display Processed Files & CSV Options --- 
    # Call the renamed and updated sidebar function
    display_processed_files_sidebar()

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Quel est votre question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Agent Logic Placeholder ---
        # This is where the agent execution logic will go later.
        # For now, let's just acknowledge the input.
        with st.chat_message("assistant"):
            response = f"Received: '{prompt}'. Agent logic will be added here." # Placeholder
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # We might need a st.rerun() here later depending on agent execution flow

if __name__ == "__main__":
    main()