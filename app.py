import streamlit as st
import os
import shutil
import uuid
import torch
from smolagents import OpenAIServerModel
from ui_components import handle_pdf_upload, handle_csv_upload, cleanup_resources
from agent_utils import (
    initialize_search_agent, 
    initialize_data_analyst_agent, 
    initialize_rag_agent, 
    route_request
)

# -------- CONFIGURATION --------
# Keep constants needed by app.py or potentially passed to utils
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Moved to agent_utils
# PDF_CLASSES = ['Select Classification...', 'finance policies', 'legal AML', 'general'] # Moved to ui_components
# -------------------------------

torch.classes.__path__ = []

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

        st.subheader("Options Fichiers")
        # Separate uploaders for clarity
        file_type_choice = st.radio("Quel type de fichier charger ?", ("Aucun", "PDF", "CSV"), horizontal=True)

        # Common options
        user_notes = st.text_area("Notes additionnelles sur le fichier", 
                                  placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur le fichier...", 
                                  height=150)
        # CSV specific option
        use_memory_limit = True # Default
        if file_type_choice == "CSV":
            use_memory_limit = st.checkbox("Limiter l'utilisation de la mémoire (CSV)", value=True)
        
    # -------------------------------------

    # --- Interface Principale ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # Initialize PDF state keys if they don't exist
        st.session_state.setdefault('uploaded_file_id', None)
        st.session_state.setdefault('pdf_summary', None)
        st.session_state.setdefault('pdf_suggested_classification', None)
        st.session_state.setdefault('pdf_classification', None)
        st.session_state.setdefault('pdf_indexed', False)
        st.session_state.setdefault('pdf_temp_path', None)
        st.session_state.setdefault('pdf_db_path', None)
        
    figures_dir = os.path.join('data', 'figures', st.session_state.session_id)

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
    
    chunk_size = 100000 if use_memory_limit else None

    # --- File Upload Handling (using functions from ui_components) ---
    csv_args = None
    if file_type_choice == "PDF":
        # Pass the initialized model, API key, and notes
        handle_pdf_upload(model, mistral_api_key, user_notes)
    elif file_type_choice == "CSV":
        # Pass relevant options
        csv_args = handle_csv_upload(user_notes, use_memory_limit, figures_dir, chunk_size)
    elif file_type_choice == "Aucun":
        # Clear potentially lingering PDF state if user switches from PDF to Aucun
        # This logic might need refinement depending on desired behavior when switching file types
        if st.session_state.get('uploaded_file_id') is not None: 
             st.session_state.uploaded_file_id = None
             st.session_state.pdf_summary = None
             st.session_state.pdf_suggested_classification = None
             st.session_state.pdf_classification = None
             st.session_state.pdf_indexed = False
             st.session_state.pdf_temp_path = None
             st.session_state.pdf_db_path = None
             pdf_temp_dir = os.path.join("data", "pdf_temp", st.session_state.session_id)
             if os.path.exists(pdf_temp_dir):
                 try:
                     shutil.rmtree(pdf_temp_dir)
                 except Exception as e:
                     st.warning(f"Could not remove temp PDF dir: {e}")

    # --- User Query Input ---
    user_query = st.text_input("Requête à envoyer à l'agent")

    # --- Main Execution Button ---
    if st.button("Exécuter"):
        # Validation checks
        if not api_key:
            st.error("Veuillez entrer une clé API OpenAI valide.")
            return
            
        # Check Mistral key only if PDF interaction is possible (classification selected)
        if st.session_state.get('pdf_classification') and not st.session_state.get('pdf_indexed'):
             # It's selected but not indexed yet (user hasn't clicked index button?)
             # Or if the RAG agent might be needed based on state
             if not mistral_api_key:
                  st.error("Clé API Mistral requise pour l'indexation ou l'interrogation PDF.")
                  return
                  
        # Allow execution if query OR csv OR indexed PDF exists
        if not user_query and not csv_args and not st.session_state.get('pdf_indexed'):
             st.error("Veuillez entrer une requête ou fournir et indexer un fichier PDF ou fournir un CSV.")
             return

        # Ensure model is initialized
        if not model:
            st.error("Modèle OpenAI non initialisé. Vérifiez la clé API.")
            return

        # Set keys in environment (may be redundant if set elsewhere, but safe)
        os.environ["OPENAI_API_KEY"] = api_key
        if mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
            
        # Cleanup old figures before execution
        cleanup_resources(figures_dir) 

        # --- Agent Initialization (using functions from agent_utils) ---
        search_agent = None
        data_analyst = None
        rag_agent = None

        with st.spinner("Initialisation des agents..."):
            search_agent = initialize_search_agent(model)
            data_analyst = initialize_data_analyst_agent(model)
            
            # Initialize RAG agent ONLY if PDF was indexed successfully
            if st.session_state.get('pdf_indexed') and st.session_state.get('pdf_db_path'):
                rag_agent = initialize_rag_agent(model, st.session_state.pdf_db_path)
                if rag_agent:
                    st.success("RAG Agent initialisé.") # Moved success message here
            
        # --- Execute Query ---
        # Proceed only if at least one agent relevant to the query is initialized
        if user_query or csv_args:
            pdf_context = None
            if st.session_state.get('pdf_indexed'):
                pdf_context = { 
                    "summary": st.session_state.get('pdf_summary'),
                    "classification": st.session_state.get('pdf_classification'),
                    "db_path": st.session_state.get('pdf_db_path')
                }
                # Optional: Display summary again if querying indexed PDF
                # with st.expander("Résumé du PDF traité"):
                #     st.markdown(st.session_state.get('pdf_summary', "Aucun résumé disponible."))

            with st.spinner("L'agent traite votre requête..."):
                try:
                    result = route_request(
                        query=user_query, 
                        csv_args=csv_args, 
                        search_agent=search_agent, 
                        data_analyst=data_analyst, 
                        rag_agent=rag_agent, 
                        pdf_context=pdf_context
                    ) 
                    st.subheader("Résultat de l'agent")
                    st.markdown(result, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erreur lors du traitement de la requête: {str(e)}")
                    # return # Optional: Stop execution on error

            # Display figures generated by Data Analyst
            if csv_args and os.path.exists(figures_dir) and os.listdir(figures_dir):
                st.subheader("Figures générées (Analyse CSV)")
                for fig_file in os.listdir(figures_dir):
                    if fig_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        st.image(os.path.join(figures_dir, fig_file), caption=fig_file)
                st.info("💡 Conseil: Vous pouvez demander à l'agent d'effectuer des analyses plus spécifiques sur vos données CSV.")
        
        elif not user_query and st.session_state.get('pdf_indexed'):
             st.info("Le fichier PDF a été indexé. Entrez une requête pour interroger son contenu.")
        
        # Fallback/info messages
        # else:
        #    st.warning("Aucune action effectuée. Veuillez fournir une requête.")

if __name__ == "__main__":
    main()