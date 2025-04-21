import streamlit as st
import os
import shutil
import uuid
from PyPDF2 import PdfReader
from utils.pipeline_indexation.pdf_processor import analyser_pdf
from utils.pipeline_indexation.csv_processor import CSVProcessor
from smolagents import OpenAIServerModel # For type hinting if needed, or pass model instance

# Define PDF_CLASSES here or pass it as an argument if preferred
PDF_CLASSES = ['Select Classification...', 'finance policies', 'legal AML', 'general']

def cleanup_resources(figures_dir):
    """
    Nettoie les ressources utilisées par les agents (figures).

    Args:
        figures_dir: Chemin vers le dossier contenant les figures générées.
    """
    if os.path.exists(figures_dir):
        try:
            shutil.rmtree(figures_dir)
            os.makedirs(figures_dir, exist_ok=True)
        except Exception as e:
            st.warning(f"Impossible de nettoyer le dossier des figures: {e}")
    else:
        os.makedirs(figures_dir, exist_ok=True)


def handle_pdf_upload(model: OpenAIServerModel, mistral_api_key: str, user_notes: str):
    """
    Handles the PDF file upload, analysis, classification selection, and indexing UI.

    Manages relevant session state variables for PDF processing.

    Args:
        model: The initialized OpenAIServerModel instance for analysis.
        mistral_api_key: The Mistral API key for indexing.
        user_notes: Additional notes provided by the user.

    Returns:
        None. Updates session state directly.
    """
    st.subheader("Chargement de Fichier PDF")
    uploaded_file = st.file_uploader("Déposer un fichier PDF", type=['pdf'], key="pdf_uploader")

    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

        # Process only if it's a new file compared to the last processed one
        if st.session_state.get('uploaded_file_id') != current_file_id:
            st.session_state.uploaded_file_id = current_file_id
            # Reset PDF state for the new file
            st.session_state.pdf_summary = None
            st.session_state.pdf_suggested_classification = None
            st.session_state.pdf_classification = None
            st.session_state.pdf_indexed = False
            st.session_state.pdf_temp_path = None
            st.session_state.pdf_db_path = None
            # Clean up previous temp dir if it exists from a *different* session/file
            pdf_temp_dir_old = os.path.join("data", "pdf_temp", st.session_state.session_id)
            if os.path.exists(pdf_temp_dir_old):
                 try: shutil.rmtree(pdf_temp_dir_old)
                 except Exception: pass


            if not model:
                st.warning("Clé API OpenAI requise pour analyser le PDF.")
            else:
                pdf_temp_dir = os.path.join("data", "pdf_temp", st.session_state.session_id)
                os.makedirs(pdf_temp_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_temp_dir, uploaded_file.name)
                try:
                    # Save the uploaded file temporarily
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state.pdf_temp_path = pdf_path

                    # 1. Extract text
                    with st.spinner(f"Extraction du texte du PDF '{uploaded_file.name}'..."):
                        reader = PdfReader(pdf_path)
                        pdf_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

                    # 2. Generate Summary and Suggest Classification (only if text was extracted)
                    if pdf_text:
                        with st.spinner("Analyse par l'IA (Résumé & Classification)..."):
                            classification_prompt = (
                                f"Analysez le texte PDF suivant et les notes de l'utilisateur. "
                                f"1. Fournissez un résumé concis (max 150 mots).\n"
                                f"2. Suggérez la classification la plus appropriée parmi : {', '.join(PDF_CLASSES[1:])}. "
                                f"Répondez UNIQUEMENT avec le résumé suivi de \\'\\nSuggested Classification: [classe]\\' à la fin.\\n\\n"
                                f"Notes de l'utilisateur: {user_notes if user_notes else 'Aucune'}\\n\\n"
                                f"Texte du PDF (premiers 5000 caractères):\\n{pdf_text[:5000]}"
                            )
                            try:
                                chat_message = model(messages=[{"role": "user", "content": classification_prompt}])
                                llm_response = chat_message.content

                                summary = llm_response
                                suggested_class = None
                                if "\nSuggested Classification:" in llm_response:
                                    parts = llm_response.split("\nSuggested Classification:")
                                    summary = parts[0].strip()
                                    suggestion_text = parts[1].strip()
                                    for cls in PDF_CLASSES[1:]:
                                        if cls.lower() == suggestion_text.lower():
                                            suggested_class = cls
                                            break
                                    if not suggested_class:
                                        print(f"Warning: LLM suggested class '{suggestion_text}' not in allowed list {PDF_CLASSES[1:]}, defaulting to general.")
                                        suggested_class = 'general'
                                else:
                                     print("Warning: Could not parse suggested classification from LLM response.")
                                     suggested_class = 'general'

                                st.session_state.pdf_summary = summary
                                st.session_state.pdf_suggested_classification = suggested_class
                                st.session_state.pdf_classification = suggested_class or PDF_CLASSES[0]

                            except Exception as llm_error:
                                st.error(f"Erreur lors de la communication avec l'IA : {llm_error}")
                                st.session_state.pdf_summary = "Erreur lors de la génération du résumé."
                    else:
                        st.warning("Aucun texte n'a pu être extrait du PDF.")
                        st.session_state.pdf_summary = "Impossible d'extraire le texte."

                except Exception as e:
                    st.error(f"Erreur lors de la préparation du PDF : {str(e)}")
                    if 'pdf_temp_dir' in locals() and os.path.exists(pdf_temp_dir):
                        try: shutil.rmtree(pdf_temp_dir)
                        except Exception: pass
                    st.session_state.uploaded_file_id = None
                    st.session_state.pdf_summary = None
                    st.session_state.pdf_suggested_classification = None
                    st.session_state.pdf_classification = None
                    st.session_state.pdf_indexed = False
                    st.session_state.pdf_temp_path = None
                    st.session_state.pdf_db_path = None

        # --- Display Summary & Classification Choice ---
        if st.session_state.get('pdf_summary'):
            st.subheader("Résumé du PDF (généré par IA)")
            st.markdown(st.session_state.pdf_summary)

            current_selection = st.session_state.get('pdf_classification', PDF_CLASSES[0])
            try:
                default_index = PDF_CLASSES.index(current_selection)
            except ValueError:
                default_index = 0

            def update_classification():
                new_selection = st.session_state.pdf_classification_selector
                if new_selection != st.session_state.get('pdf_classification'):
                    st.session_state.pdf_classification = new_selection if new_selection != PDF_CLASSES[0] else None
                    st.session_state.pdf_indexed = False
                    st.session_state.pdf_db_path = None
                    print(f"Classification selection changed to: {st.session_state.pdf_classification}")

            selected_class = st.selectbox(
                "Confirmez ou modifiez la classification de ce document :",
                PDF_CLASSES,
                key='pdf_classification_selector',
                index=default_index,
                on_change=update_classification
            )

            disable_button = not st.session_state.get('pdf_classification') or st.session_state.get('pdf_indexed')
            button_text = "Confirmer la classification et indexer"
            if st.session_state.get('pdf_indexed') and st.session_state.get('pdf_classification') == selected_class:
                 button_text = f"Déjà indexé comme '{selected_class}'"

            if st.button(button_text, key="index_pdf_button", disabled=disable_button):
                if st.session_state.pdf_classification:
                     st.info(f"Classification '{st.session_state.pdf_classification}' sélectionnée. Tentative d'indexation...")
                     if not mistral_api_key:
                         st.error("Clé API Mistral requise pour l'indexation PDF.")
                     elif not st.session_state.get('pdf_temp_path') or not os.path.exists(st.session_state.pdf_temp_path):
                         st.error("Erreur : Chemin du fichier PDF temporaire introuvable pour l'indexation.")
                     else:
                         try:
                             with st.spinner(f"Indexation du PDF sous '{st.session_state.pdf_classification}' en cours..."):
                                 os.environ["MISTRAL_API_KEY"] = mistral_api_key
                                 base_db_dir = os.path.join("data", "output", "vectordb", st.session_state.pdf_classification)
                                 os.makedirs(base_db_dir, exist_ok=True)
                                 db_path = os.path.join(base_db_dir, f"session_{st.session_state.session_id}")
                                 print(f"Indexing PDF. Using DB path: {db_path}")
                                 analysis_summary = analyser_pdf(st.session_state.pdf_temp_path, db_path=db_path, exporter=False)
                                 st.session_state.pdf_db_path = db_path
                                 st.session_state.pdf_indexed = True
                                 st.success(f"PDF indexé avec succès dans la catégorie '{st.session_state.pdf_classification}'.")

                                 # --- Cleanup Temporary PDF ---
                                 try:
                                     if st.session_state.get('pdf_temp_path') and os.path.exists(st.session_state.pdf_temp_path):
                                         os.remove(st.session_state.pdf_temp_path)
                                         print(f"Removed temporary PDF: {st.session_state.pdf_temp_path}")
                                 except Exception as cleanup_error:
                                     st.warning(f"Could not remove temporary PDF file: {cleanup_error}")
                                 # ---------------------------
                                 st.rerun()

                         except Exception as index_error:
                             st.error(f"Erreur lors de l'indexation du PDF : {index_error}")
                             st.session_state.pdf_indexed = False
                             st.session_state.pdf_db_path = None

    elif uploaded_file is None and st.session_state.get('uploaded_file_id') is not None:
        # Clear PDF state if file is removed
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
                print(f"Removed temporary PDF directory: {pdf_temp_dir}")
            except Exception as e:
                st.warning(f"Could not remove temporary PDF directory {pdf_temp_dir}: {e}")

def handle_csv_upload(user_notes: str, use_memory_limit: bool, figures_dir: str, chunk_size: int | None):
    """
    Handles the CSV file upload, validation, analysis, and returns arguments for the Data Analyst agent.

    Args:
        user_notes: Additional notes provided by the user.
        use_memory_limit: Boolean indicating whether to use chunking.
        figures_dir: Path to the directory for saving figures.
        chunk_size: The chunk size to use if memory limit is enabled.

    Returns:
        dict | None: A dictionary containing arguments for the data_analyst agent
                     if a valid CSV is processed, otherwise None.
    """
    st.subheader("Chargement de Fichier CSV")
    uploaded_file = st.file_uploader("Déposer un fichier CSV", type=['csv'], key="csv_uploader")

    csv_args = None
    if uploaded_file is not None:
        try:
            # Create directory for CSV files if it doesn't exist
            csv_dir = os.path.join("data", "csv_files")
            os.makedirs(csv_dir, exist_ok=True)

            # Generate unique filename
            original_filename = uploaded_file.name
            base_name = os.path.splitext(original_filename)[0]
            unique_filename = f"{base_name}_{str(uuid.uuid4())[:8]}.csv"
            csv_file_path = os.path.join(csv_dir, unique_filename)

            # Read and save file content with encoding detection
            try:
                file_content = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_content = uploaded_file.getvalue().decode("latin-1")
                except UnicodeDecodeError:
                    file_content = uploaded_file.getvalue().decode("iso-8859-1", errors="replace")

            with open(csv_file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            # Use CSVProcessor for validation and analysis
            csv_processor = CSVProcessor()
            is_valid, validation_message = csv_processor.validate_csv(csv_file_path, auto_detect=True)

            if not is_valid:
                st.error(f"Le fichier CSV n'est pas valide: {validation_message}")
                # Clean up invalid file? Optional.
                # os.remove(csv_file_path)
                return None # Stop processing if invalid

            separator = csv_processor.separator
            encoding = csv_processor.encoding
            basic_analysis = csv_processor.analyze_csv(csv_file_path, auto_detect=True)
            columns = basic_analysis.get("colonnes", [])
            rows = basic_analysis.get("nombre_lignes", 0)

            # Memory warning logic
            file_size_mb = os.path.getsize(csv_file_path) / (1024 * 1024)
            memory_warning = ""
            if file_size_mb > 10 and not use_memory_limit:
                memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."

            # Prepare notes for data_analyst
            data_analyst_notes = f"""
# Guide d'analyse de données
- Fichier: {original_filename} (Encodage détecté: {encoding})
- Chemin: {csv_file_path}
- Séparateur: '{separator}'
- Dossier pour figures: '{figures_dir}'
- Taille du fichier: {file_size_mb:.2f} MB
- Nombre de lignes: {rows}
- Nombre de colonnes: {len(columns)}

# Colonnes détectées:
{', '.join(columns)}

# Notes de l'utilisateur:
{user_notes if user_notes else "Aucune note spécifique fournie par l'utilisateur."}

# Étapes recommandées:
1. Charger le CSV avec le bon séparateur: pd.read_csv('{csv_file_path}', sep='{separator}')
2. Validation du fichier CSV via csv_analyzer (déjà effectuée: {validation_message})
3. Exploration des données avec df.info(), df.describe() et vérification des valeurs manquantes
4. Création de visualisations adaptées au type de données
5. Enregistrer toutes les figures dans le dossier '{figures_dir}'
            """

            csv_args = {
                "source_file": csv_file_path,
                "separator": separator,
                "additional_notes": data_analyst_notes,
                "figures_dir": figures_dir,
                "chunk_size": chunk_size
            }

            info_message = f"Fichier CSV '{original_filename}' validé avec séparateur '{separator}'. L'agent Data Analyst sera utilisé."
            if memory_warning:
                info_message += f" {memory_warning}"
            st.info(info_message)

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier CSV: {str(e)}")
            # Clean up file if processing failed? Optional.
            # if 'csv_file_path' in locals() and os.path.exists(csv_file_path):
            #     os.remove(csv_file_path)
            return None # Return None on error

    return csv_args # Return None if no file uploaded or args if successful 