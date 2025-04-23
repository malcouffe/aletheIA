import streamlit as st
import os
import shutil
import uuid
import torch
from smolagents import OpenAIServerModel
# Import necessary functions - remove old names
from ui_components import cleanup_resources, handle_uploaded_file, index_pdf
# Keep PDF_CLASSES needed for selectbox logic moved here
import ui_components 
from agent_utils import (
    initialize_search_agent, 
    initialize_data_analyst_agent, 
    initialize_rag_agent, 
    initialize_manager_agent
)

torch.classes.__path__ = []

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

    st.subheader(f"Classifier/Indexer: {file_details.get('filename', selected_file_id)}")
    st.markdown("**Résumé du PDF (généré par IA):**")
    st.markdown(file_details.get('summary', "*Aucun résumé disponible.*"))

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
                 index_pdf(
                     file_id=selected_file_id,
                     session_id=st.session_state.session_id,
                     mistral_api_key=mistral_api_key
                 )
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
    """Attempts to clean up resources associated with a single file (temp files, DBs)."""
    print(f"Attempting to cleanup resources for file_id: {file_id}")
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
                         if not os.listdir(session_dir):
                             os.rmdir(session_dir)
                             print(f"Removed empty session temp directory: {session_dir}")
                     except OSError as e:
                         print(f"Could not remove directory {temp_dir} (maybe not empty or permissions?): {e}")
            elif os.path.isdir(temp_path): # Should not happen based on current logic, but check
                shutil.rmtree(temp_path)
                print(f"Removed temp directory (unexpectedly a dir): {temp_path}")
        except Exception as e:
            st.warning(f"Error removing temp resource {temp_path}: {e}")

    # Cleanup vector database
    db_path = details.get('db_path')
    if db_path and os.path.exists(db_path):
        try:
            # DB path is expected to be a directory containing index files
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
                print(f"Removed vector DB directory: {db_path}")
                class_dir = os.path.dirname(db_path)
                try:
                    if not os.listdir(class_dir):
                        os.rmdir(class_dir)
                        print(f"Removed empty classification DB directory: {class_dir}")
                except OSError as e:
                    print(f"Could not remove directory {class_dir} (maybe not empty or permissions?): {e}")
            else:
                 print(f"Warning: Expected DB path {db_path} to be a directory, but it is not.")
        except Exception as e:
            st.warning(f"Error removing vector DB {db_path}: {e}")

    # Cleanup figures? (Currently figures_dir is session-wide, not per-file)
    # If figures become per-file, add cleanup logic here.

def _delete_file_callback(file_id_to_delete: str):
    """Callback function to handle file deletion."""
    print(f"Delete button clicked for file_id: {file_id_to_delete}")
    if 'processed_files' in st.session_state and file_id_to_delete in st.session_state.processed_files:
        details = st.session_state.processed_files[file_id_to_delete]
        _cleanup_single_file_resources(file_id_to_delete, details)
        del st.session_state.processed_files[file_id_to_delete]
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
                'error_indexing_missing_temp': (st.error, "Erreur Indexation (temp manquant)"),
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
    print(f"PDF selected for action: {file_id}")

def _update_classification_callback(file_id: str):
    """Updates the classification for the selected PDF file in session state."""
    new_classification = st.session_state[f"pdf_classification_{file_id}"]
    current_details = st.session_state.processed_files[file_id]
    
    print(f"Updating classification for {file_id} to {new_classification}")
    
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
        print(f"Classification changed for {file_id}. Indexing status reset.")

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

    st.title("Agent Web avec Streamlit")
    st.write("Entrez votre clé API OpenAI et votre requête pour interroger l'agent.")

    # --- Configuration de la Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        api_key_from_env = os.environ.get("OPENAI_API_KEY")
        mistral_api_key_from_env = os.environ.get("MISTRAL_API_KEY")

        if not api_key_from_env:
            api_key = st.text_input("Clé API OpenAI", type="password")
        else:
            st.success("Clé API OpenAI trouvée")
            api_key = api_key_from_env
            
        if not mistral_api_key_from_env:
             mistral_api_key = st.text_input("Clé API Mistral (pour PDF)", type="password")
        else:
            st.success("Clé API Mistral trouvée")
            mistral_api_key = mistral_api_key_from_env

        st.subheader("Chargement de Fichier")
        uploaded_file = st.file_uploader(
            "Déposer un fichier (PDF, CSV, TXT)", 
            type=['pdf', 'csv', 'txt'], 
            key="unified_uploader"
        )

        user_notes = st.text_area("Notes additionnelles sur le fichier",
                                  placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur le fichier...",
                                  height=150)
        # CSV specific option
        use_memory_limit = True

    # -------------------------------------

    # --- Interface Principale ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = {}

    figures_dir = os.path.join('data', 'figures', st.session_state.session_id) # TODO: Should figures be per-file?
    os.makedirs(figures_dir, exist_ok=True)

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
        temp_chunk_size = 100000 if use_memory_limit else None # Keep this logic? Or handle in handler?

        processed_file_info = handle_uploaded_file(
            uploaded_file=uploaded_file,
            session_id=st.session_state.session_id,
            model=model,
            mistral_api_key=mistral_api_key,
            user_notes=user_notes,
            use_memory_limit=use_memory_limit,
            figures_dir=figures_dir, # Pass figures dir (consider if per-file needed later)
        )

        if processed_file_info and 'file_id' in processed_file_info and 'details' in processed_file_info:
            file_id = processed_file_info['file_id']
            details = processed_file_info['details']
            st.session_state.processed_files[file_id] = details
            st.success(f"Fichier '{details.get('filename', 'Inconnu')}' traité et ajouté à la session.")
            # No longer clearing uploader via flag

            # --- Auto-select PDF for action if just uploaded --- 
            if details.get('type') == 'pdf' and details.get('status') == 'awaiting_classification':
                st.session_state.selected_file_id_for_action = file_id
                print(f"Auto-selected newly uploaded PDF {file_id} for action.")
                st.rerun()
            # ----------------------------------------------------

        elif processed_file_info:
            st.warning("Le traitement du fichier a terminé mais n'a pas pu être ajouté à l'état de la session.")

    # --- PDF Classification/Indexing UI (Conditional) ---
    # TODO: Refactor this section significantly for multi-file
    # Needs to be driven by selection from the sidebar, showing UI for one selected PDF
    display_pdf_action_section(mistral_api_key)

    # --- Display Processed Files & CSV Options --- 
    # Call the renamed and updated sidebar function
    display_processed_files_sidebar()

    # --- User Query Input ---
    user_query = st.text_input("Requête à envoyer à l'agent")

    # --- Main Execution Button ---
    if st.button("Exécuter"):
        if not api_key:
            st.error("Veuillez entrer une clé API OpenAI valide.")
            return

        if st.session_state.get('pdf_classification') and not st.session_state.get('pdf_indexed'):
             if not mistral_api_key:
                  st.error("Clé API Mistral requise pour l'indexation PDF.")
                  return

        chunk_size = None
        if st.session_state.get('csv_args'):
            use_memory_limit = st.session_state.csv_memory_limit_active
            chunk_size = 100000 if use_memory_limit else None
            if st.session_state.csv_args.get('chunk_size') != chunk_size:
                 st.session_state.csv_args['chunk_size'] = chunk_size
                 print(f"Updated chunk_size for active CSV to: {chunk_size}")

        csv_args = st.session_state.get('csv_args')
        pdf_indexed = st.session_state.get('pdf_indexed')

        if not user_query and not csv_args and not pdf_indexed:
             st.error("Veuillez entrer une requête ou fournir et traiter un fichier (PDF ou CSV).")
             return

        if not model:
            st.error("Modèle OpenAI non initialisé. Vérifiez la clé API.")
            return

        os.environ["OPENAI_API_KEY"] = api_key
        if mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key

        cleanup_resources(figures_dir)

        search_agent = None
        data_analyst = None
        rag_agent = None

        with st.spinner("Initialisation des agents..."):
            search_agent = initialize_search_agent(model)
            data_analyst = initialize_data_analyst_agent(model)
            manager_agent = initialize_manager_agent(model)

            if pdf_indexed and st.session_state.get('pdf_db_path'):
                rag_agent = initialize_rag_agent(model, st.session_state.pdf_db_path)
                if rag_agent:
                    st.success("RAG Agent initialisé.")

        if user_query or csv_args:
            pdf_context = None
            if pdf_indexed:
                pdf_context = {
                    "summary": st.session_state.get('pdf_summary'),
                    "classification": st.session_state.get('pdf_classification'),
                    "db_path": st.session_state.get('pdf_db_path'),
                    "filename": st.session_state.get('pdf_filename'),
                    "user_notes": user_notes
                }

            with st.spinner("L'agent traite votre requête..."):
                manager_prompt = build_manager_prompt(user_query, csv_args, pdf_context)

                manager_additional_args = {
                    "search_agent": search_agent,
                    "data_analyst": data_analyst,
                    "rag_agent": rag_agent,
                    "csv_args": csv_args,
                    "pdf_context": pdf_context
                }

                try:
                    result = manager_agent.run(
                        manager_prompt,
                        additional_args=manager_additional_args
                    )
                    st.subheader("Résultat de l'agent")
                    st.markdown(result, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erreur lors de l'exécution du manager agent: {str(e)}")
                    st.exception(e)

            if st.session_state.get('csv_args') and os.path.exists(figures_dir) and os.listdir(figures_dir):
                st.subheader("Figures générées (Analyse CSV)")
                for fig_file in os.listdir(figures_dir):
                    if fig_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        st.image(os.path.join(figures_dir, fig_file), caption=fig_file)
                st.info("💡 Conseil: Vous pouvez demander à l'agent d'effectuer des analyses plus spécifiques sur vos données CSV.")

        elif not user_query and pdf_indexed:
             st.info("Le fichier PDF a été indexé. Entrez une requête pour interroger son contenu.")

if __name__ == "__main__":
    main()