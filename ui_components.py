import streamlit as st
import os
import shutil
import uuid
from PyPDF2 import PdfReader
from utils.pipeline_indexation.pdf_processor import analyser_pdf
from utils.pipeline_indexation.csv_processor import CSVProcessor
from smolagents import OpenAIServerModel
import mimetypes

# Define PDF_CLASSES here or pass it as an argument if preferred
PDF_CLASSES = ['Select classification...', 'Tokenisation', 'AML-CFT', 'General']

def handle_uploaded_file(
    uploaded_file,
    session_id: str,
    model: OpenAIServerModel | None,
    mistral_api_key: str,
    use_memory_limit: bool,
    ):
    """
    Detects the type of the uploaded file, processes it, and returns its details.
    Does NOT set session state directly. Creates unique paths using session_id and a new file_id.

    Returns:
        dict | None: {'file_id': str, 'details': dict} on success, None on failure.
    """
    # --- Check if this specific file object has already been processed ---
    if uploaded_file:
        # Use name, size, and internal file_id for a robust key against reruns
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        if st.session_state.get('last_processed_file_key') == current_file_key:
            print(f"Skipping re-processing for already processed file object: {current_file_key}")
            return None # Don't process the same object again
    else:
        # Should not happen if called from app.py's check, but good practice
        return None
    # ------------------------------------------------------------------

    file_type = None
    file_name = uploaded_file.name
    file_id = str(uuid.uuid4()) # Generate unique ID for storing details

    # Use mimetype first, fallback to extension
    mime_type, _ = mimetypes.guess_type(file_name)
    print(f"Uploaded file: {file_name}, MIME type: {mime_type}, Streamlit type: {uploaded_file.type}, Assigned file_id: {file_id}")

    if uploaded_file.type == "application/pdf" or (mime_type == "application/pdf") or file_name.lower().endswith('.pdf'):
        file_type = "pdf"
    elif uploaded_file.type == "text/csv" or (mime_type == "text/csv") or file_name.lower().endswith('.csv'):
        file_type = "csv"
    elif uploaded_file.type == "text/plain" or (mime_type == "text/plain") or file_name.lower().endswith('.txt'):
        file_type = "txt"
    else:
        # Attempt guess based on extension if type is generic (like application/octet-stream)
         if file_name.lower().endswith('.pdf'): file_type = "pdf"
         elif file_name.lower().endswith('.csv'): file_type = "csv"
         elif file_name.lower().endswith('.txt'): file_type = "txt"

    details = None

    if file_type == "pdf":
        st.info(f"Fichier PDF '{file_name}' (ID: {file_id}) détecté. Traitement en cours...")
        details = _handle_pdf_logic(
            file_id=file_id,
            uploaded_file=uploaded_file,
            session_id=session_id,
            model=model,
            use_memory_limit=use_memory_limit,
        )
    elif file_type == "csv":
        st.info(f"Fichier CSV '{file_name}' (ID: {file_id}) détecté. Traitement en cours...")
        chunk_size = 100000 if use_memory_limit else None
        details = _handle_csv_logic(
            file_id=file_id,
            uploaded_file=uploaded_file,
            session_id=session_id,
            use_memory_limit=use_memory_limit,
            chunk_size=chunk_size
        )
    elif file_type == "txt":
        st.warning(f"Fichier TXT '{file_name}' détecté. L'analyse de fichiers TXT n'est pas encore implémentée.")
        # Return None as it's not handled
        return None
    else:
        st.error(f"Type de fichier non supporté ou non détecté pour '{file_name}' (Type MIME: {mime_type}, Type Streamlit: {uploaded_file.type}). Veuillez charger un fichier PDF, CSV ou TXT.")
        return None

    # If processing was successful, add common details and return structured dict
    if details:
        details['file_id'] = file_id
        details['filename'] = file_name
        details['type'] = file_type

        # Store the key of the file object we just successfully processed
        st.session_state.last_processed_file_key = current_file_key

        return {'file_id': file_id, 'details': details}
    else:
        # Processing failed, handler should have shown an error
        return None

def _handle_pdf_logic(
    file_id: str,
    uploaded_file,
    session_id: str,
    model: OpenAIServerModel | None,
    use_memory_limit: bool,
    ):
    """
    Internal logic to handle PDF processing (analysis, classification suggestion).
    Returns a dictionary with processing results or None on failure.
    Does NOT set session state.
    """
    if not model:
        st.warning("Clé API OpenAI requise pour analyser le PDF.")
        return None

    # Create specific temp directory using session_id and file_id
    pdf_temp_dir = os.path.join("data", "pdf_temp", session_id, file_id)
    os.makedirs(pdf_temp_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_temp_dir, uploaded_file.name)

    try:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner(f"Extraction du texte du PDF '{uploaded_file.name}'..."):
            reader = PdfReader(pdf_path)
            pdf_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

        if pdf_text:
            with st.spinner("Analyse par l'IA (Résumé & Classification)..."):
                classification_prompt = (
                    f"Analysez le texte PDF suivant. "
                    f"1. Fournissez un résumé concis (max 150 mots).\n"
                    f"2. Suggérez la classification la plus appropriée parmi : {', '.join(PDF_CLASSES[1:])}. "
                    f"Répondez UNIQUEMENT avec le résumé suivi de '\nSuggested Classification: [classe]' à la fin.\n\n"
                    f"Texte du PDF (premiers 5000 caractères):\n{pdf_text[:5000]}"
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
                            # Case-insensitive comparison
                            if cls.lower() == suggestion_text.lower():
                                suggested_class = cls
                                break
                        if not suggested_class:
                            print(f"Warning: LLM suggested class '{suggestion_text}' not in allowed list {PDF_CLASSES[1:]}, defaulting to 'general'.")
                            suggested_class = 'general' # Default to general if suggestion is weird
                    else:
                            print("Warning: Could not parse suggested classification from LLM response. Defaulting to 'general'.")
                            suggested_class = 'general' # Default if format is wrong

                    # Return details instead of setting state
                    pdf_details = {
                        'summary': summary,
                        'suggested_classification': suggested_class,
                        'classification': None,
                        'temp_path': pdf_path,
                        'status': 'awaiting_classification',
                        'indexed': False,
                        'db_path': None
                    }
                    return pdf_details

                except Exception as llm_error:
                    st.error(f"Erreur lors de la communication avec l'IA : {llm_error}")
                    return None
        else:
            st.warning("Aucun texte n'a pu être extrait du PDF.")
            # Return minimal info indicating failure to extract text but file exists
            return {'summary': "Impossible d'extraire le texte.", 'temp_path': pdf_path, 'status': 'error_extraction'}

    except Exception as e:
        st.error(f"Erreur lors de la préparation du PDF : {str(e)}")
        # Ensure temp dir is removed if it exists but processing failed fundamentally
        if os.path.exists(pdf_temp_dir):
             try: shutil.rmtree(pdf_temp_dir)
             except Exception as rm_err: print(f"Error cleaning up failed PDF temp dir {pdf_temp_dir}: {rm_err}")
        return None

def index_pdf(file_id: str, file_details: dict, session_id: str, mistral_api_key: str) -> dict | None:
     """
     Performs the PDF indexing using the provided file details and session ID.
     Returns an updated dictionary with indexing results or None on failure.
     Does NOT modify session state directly, but uses st functions for user feedback.

     Args:
         file_id (str): The unique ID of the file being processed.
         file_details (dict): The dictionary containing details about the file.
         session_id (str): The ID of the current user session.
         mistral_api_key (str): The Mistral API key for indexing.

     Returns:
         dict | None: An updated file details dictionary on success/error, or None on critical setup failure.
     """
     # Create a copy to avoid modifying the original dict directly before returning
     updated_details = file_details.copy()

     # Check current status and required info from the input details
     if updated_details.get('status') != 'awaiting_classification' and updated_details.get('status') != 'classified':
          st.warning(f"Cannot index PDF '{updated_details.get('filename')}'. Status is '{updated_details.get('status')}'.")
          # Return None or original details? Let's return original details to signify no state change attempted.
          # Or return None? Returning details seems better.
          return updated_details
     if updated_details.get('indexed'):
         st.warning(f"PDF '{updated_details.get('filename')}' is already indexed.")
         return updated_details
     if not updated_details.get('classification'):
          st.warning(f"Cannot index PDF '{updated_details.get('filename')}'. No classification selected.")
          return updated_details
     if not mistral_api_key:
         st.error("Clé API Mistral requise pour l'indexation PDF.")
         # This is a configuration error, perhaps return None? Or mark as error state?
         updated_details['status'] = 'error_missing_api_key'
         return updated_details
     pdf_temp_path = updated_details.get('temp_path')
     if not pdf_temp_path or not os.path.exists(pdf_temp_path):
         st.error(f"Erreur : Chemin du fichier PDF temporaire '{pdf_temp_path}' introuvable pour l'indexation du fichier ID '{file_id}'.")
         updated_details.update({
             'indexed': False,
             'db_path': None,
             'status': 'error_indexing_missing_temp'
         })
         return updated_details

     classification = updated_details['classification']
     filename = updated_details.get('filename', f'ID: {file_id}') # Use filename or ID for messages
     st.info(f"Classification '{classification}' sélectionnée pour '{filename}'. Tentative d'indexation...")

     try:
         with st.spinner(f"Indexation du PDF '{filename}' (ID: {file_id}) sous '{classification}' en cours..."):
             os.environ["MISTRAL_API_KEY"] = mistral_api_key

             # Create DB path incorporating session_id and file_id for uniqueness
             base_db_dir = os.path.join("data", "output", "vectordb", classification)
             db_path = os.path.join(base_db_dir, f"session_{session_id}_file_{file_id}")
             os.makedirs(os.path.dirname(db_path), exist_ok=True)

             print(f"Indexing PDF. File ID: {file_id}, Temp Path: {pdf_temp_path}, DB path: {db_path}")
             # Assuming analyser_pdf takes path and db_path
             analyser_pdf(pdf_temp_path, db_path=db_path, exporter=False)

             updated_details.update({
                 'db_path': db_path,
                 'indexed': True,
                 'status': 'indexed',
                 'temp_path': None
             })
             st.success(f"Fichier '{filename}' (ID: {file_id}) indexé avec succès dans la catégorie '{classification}'.")

             # --- Cleanup Temporary PDF ---
             try:
                 if os.path.exists(pdf_temp_path):
                     os.remove(pdf_temp_path)
                     print(f"Removed temporary PDF: {pdf_temp_path}")
             except Exception as cleanup_error:
                 st.warning(f"Could not remove temporary PDF file '{pdf_temp_path}': {cleanup_error}")
             # ---------------------------

             return updated_details

     except Exception as index_error:
         st.error(f"Erreur lors de l'indexation du PDF '{filename}' (ID: {file_id}): {index_error}")
         updated_details.update({
             'indexed': False,
             'db_path': None,
             'status': 'error_indexing'
         })
         return updated_details

def _handle_csv_logic(
    file_id: str,
    uploaded_file,
    session_id: str,
    use_memory_limit: bool,
    chunk_size: int | None
    ):
    """
    Internal logic to handle CSV processing (validation, analysis prep).
    Returns a dictionary with processing results or None on failure.
    Does NOT set session state.
    """
    csv_args = None
    csv_file_path = None

    try:
        # Create specific temp directory using session_id and file_id
        csv_temp_dir = os.path.join("data", "csv_temp", session_id, file_id)
        os.makedirs(csv_temp_dir, exist_ok=True)

        original_filename = uploaded_file.name
        # Keep original filename for reference, save with original name in its unique dir
        csv_file_path = os.path.join(csv_temp_dir, original_filename)

        try:
            # Decode with fallback
            file_content = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                file_content = uploaded_file.getvalue().decode("latin-1")
            except UnicodeDecodeError:
                file_content = uploaded_file.getvalue().decode("iso-8859-1", errors="replace")

        with open(csv_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        csv_processor = CSVProcessor()
        is_valid, validation_message = csv_processor.validate_csv(csv_file_path, auto_detect=True)

        if not is_valid:
            st.error(f"Le fichier CSV '{original_filename}' (ID: {file_id}) n'est pas valide: {validation_message}")
            if os.path.exists(csv_file_path):
                try: os.remove(csv_file_path)
                except Exception as rm_err: print(f"Error removing invalid CSV {csv_file_path}: {rm_err}")
            try: os.rmdir(csv_temp_dir)
            except Exception as rmdir_err: print(f"Error removing empty temp dir {csv_temp_dir}: {rmdir_err}")
            return None

        separator = csv_processor.separator
        encoding = csv_processor.encoding
        basic_analysis = csv_processor.analyze_csv(csv_file_path, auto_detect=True)
        columns = basic_analysis.get("colonnes", [])
        rows = basic_analysis.get("nombre_lignes", 0)

        file_size_mb = os.path.getsize(csv_file_path) / (1024 * 1024)
        memory_warning = ""
        if file_size_mb > 10 and not use_memory_limit:
            memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."
            # Note: The checkbox state is managed in app.py, this is just informational

        # Define the per-file figures directory
        figures_dir = os.path.join('data', 'figures', session_id, file_id)
        os.makedirs(figures_dir, exist_ok=True)

        # Update path in notes
        data_analyst_notes = f"""
# Guide d'analyse de données
- Fichier: {original_filename} (Encodage détecté: {encoding})
- File ID: {file_id}
- Chemin: {csv_file_path}
- Séparateur: '{separator}'
- Dossier pour figures: '{figures_dir}'
- Taille du fichier: {file_size_mb:.2f} MB
- Nombre de lignes: {rows}
- Nombre de colonnes: {len(columns)}

# Colonnes détectées:
{', '.join(columns)}

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

        info_message = f"Fichier CSV '{original_filename}' validé avec séparateur '{separator}'. Prêt pour l'analyse."
        if memory_warning:
            info_message += f" {memory_warning}"
        st.success(info_message)

        # Return details
        csv_details = {
            'csv_args': csv_args,
            'temp_path': csv_file_path,
            'status': 'ready',
            'columns': columns,
            'rows': rows,
            'size_mb': file_size_mb
        }
        return csv_details

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier CSV '{uploaded_file.name}' (ID: {file_id}): {str(e)}")
        # Clean up temp file/dir on error
        if csv_file_path and os.path.exists(csv_file_path):
            try: os.remove(csv_file_path)
            except Exception as rm_err: print(f"Error removing CSV {csv_file_path} on error: {rm_err}")
        if 'csv_temp_dir' in locals() and os.path.exists(csv_temp_dir):
             try: shutil.rmtree(csv_temp_dir)
             except Exception as rmdir_err: print(f"Error removing temp dir {csv_temp_dir} on error: {rmdir_err}")
        return None