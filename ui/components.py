import streamlit as st
import os
import shutil
import uuid
from PyPDF2 import PdfReader
from utils.pipeline_indexation.pdf_processor import analyser_pdf
from utils.pipeline_indexation.csv_processor import CSVProcessor
from smolagents import OpenAIServerModel
import mimetypes
import time

# Define PDF_CLASSES here or pass it as an argument if preferred
PDF_CLASSES = ['Select Classification...', 'finance policies', 'legal AML', 'general']

def handle_uploaded_file(
    uploaded_file,
    session_id: str,
    model: OpenAIServerModel | None,
    mistral_api_key: str,
    use_memory_limit: bool,
    ):
    """
    Detects file type, processes it, and returns details.
    Returns: dict | None
    """
    if uploaded_file:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        if st.session_state.get('last_processed_file_key') == current_file_key:
            print(f"Skipping re-processing for already processed file object: {current_file_key}")
            return {'already_processed': True, 'file_key': current_file_key}
    else:
        return None

    file_name = uploaded_file.name
    file_id = str(uuid.uuid4())
    mime_type, _ = mimetypes.guess_type(file_name)
    print(f"Uploaded file: {file_name}, MIME type: {mime_type}, Streamlit type: {uploaded_file.type}, Assigned file_id: {file_id}")

    # Determine file type
    if uploaded_file.type == "application/pdf" or (mime_type == "application/pdf") or file_name.lower().endswith('.pdf'):
        file_type = "pdf"
    elif uploaded_file.type == "text/csv" or (mime_type == "text/csv") or file_name.lower().endswith('.csv'):
        file_type = "csv"
    elif uploaded_file.type == "text/plain" or (mime_type == "text/plain") or file_name.lower().endswith('.txt'):
        file_type = "txt"
    else:
        if file_name.lower().endswith('.pdf'): 
            file_type = "pdf"
        elif file_name.lower().endswith('.csv'): 
            file_type = "csv"
        elif file_name.lower().endswith('.txt'): 
            file_type = "txt"
        else:
            file_type = None

    # Process based on file type
    if file_type == "pdf":
        st.info(f"Fichier PDF '{file_name}' détecté. Extraction et analyse en cours...")
        details = _handle_pdf_logic(file_id, uploaded_file, session_id, model, use_memory_limit)
    elif file_type == "csv":
        st.info(f"Fichier CSV '{file_name}' détecté. Validation et analyse en cours...")
        chunk_size = 100000 if use_memory_limit else None
        details = _handle_csv_logic(file_id, uploaded_file, session_id, use_memory_limit, chunk_size)
    elif file_type == "txt":
        st.warning(f"Fichier TXT '{file_name}' détecté. L'analyse de fichiers TXT n'est pas encore implémentée.")
        return None
    else:
        st.error(f"Type de fichier non supporté ou non détecté pour '{file_name}' (Type MIME: {mime_type}, Type Streamlit: {uploaded_file.type}). Veuillez charger un fichier PDF, CSV ou TXT.")
        return None

    if details:
        details.update({
            'file_id': file_id,
            'filename': file_name,
            'type': file_type
        })
        st.session_state.last_processed_file_key = current_file_key
        return {'file_id': file_id, 'details': details}
    else:
        return None

def _handle_pdf_logic(file_id: str, uploaded_file, session_id: str, model: OpenAIServerModel | None, use_memory_limit: bool):
    """Handle PDF processing and return details dict."""
    if not model:
        st.warning("Clé API OpenAI requise pour analyser le PDF.")
        return None

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
                    suggested_class = 'general'
                    
                    if "\nSuggested Classification:" in llm_response:
                        parts = llm_response.split("\nSuggested Classification:")
                        summary = parts[0].strip()
                        suggestion_text = parts[1].strip()
                        for cls in PDF_CLASSES[1:]:
                            if cls.lower() == suggestion_text.lower():
                                suggested_class = cls
                                break

                    return {
                        'summary': summary,
                        'suggested_classification': suggested_class,
                        'classification': None,
                        'temp_path': pdf_path,
                        'status': 'awaiting_classification',
                        'indexed': False,
                        'db_path': None
                    }

                except Exception as llm_error:
                    st.error(f"Erreur lors de la communication avec l'IA : {llm_error}")
                    return None
        else:
            st.warning("Aucun texte n'a pu être extrait du PDF.")
            return {'summary': "Impossible d'extraire le texte.", 'temp_path': pdf_path, 'status': 'error_extraction'}

    except Exception as e:
        st.error(f"Erreur lors de la préparation du PDF : {str(e)}")
        if os.path.exists(pdf_temp_dir):
            try: 
                shutil.rmtree(pdf_temp_dir)
            except Exception as rm_err: 
                print(f"Error cleaning up failed PDF temp dir {pdf_temp_dir}: {rm_err}")
        return None

def index_pdf(file_id: str, file_details: dict, session_id: str, mistral_api_key: str) -> dict | None:
    """Performs PDF indexing and returns updated details."""
    updated_details = file_details.copy()

    # Validation checks
    if updated_details.get('status') not in ['awaiting_classification', 'classified', 'indexing']:
        st.warning(f"Cannot index PDF '{updated_details.get('filename')}'. Status is '{updated_details.get('status')}'.")
        return updated_details
    if updated_details.get('indexed'):
        st.warning(f"PDF '{updated_details.get('filename')}' is already indexed.")
        return updated_details
    if not updated_details.get('classification'):
        st.warning(f"Cannot index PDF '{updated_details.get('filename')}'. No classification selected.")
        return updated_details
    if not mistral_api_key:
        st.error("Clé API Mistral requise pour l'indexation PDF.")
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
    filename = updated_details.get('filename', f'ID: {file_id}')
    
    # Immediately update status to indexing
    updated_details['status'] = 'indexing'

    try:
        # Create progress container with minimal UI updates
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Setup - NO chat spam
            status_text.text("🔧 Configuration de l'environnement...")
            progress_bar.progress(10)
            os.environ["MISTRAL_API_KEY"] = mistral_api_key

            base_db_dir = os.path.join("data", "output", "vectordb", classification)
            db_path = os.path.join(base_db_dir, f"session_{session_id}_file_{file_id}")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Step 2: Start indexing - NO chat spam
            status_text.text("📄 Analyse et indexation du PDF en cours...")
            progress_bar.progress(50)
            print(f"Indexing PDF. File ID: {file_id}, Temp Path: {pdf_temp_path}, DB path: {db_path}")
            
            # Perform the actual PDF analysis
            analyser_pdf(pdf_temp_path, db_path=db_path, exporter=False)
            
            # Step 3: Finalization - NO chat spam
            progress_bar.progress(90)
            status_text.text("✅ Finalisation de l'indexation...")

            updated_details.update({
                'db_path': db_path,
                'indexed': True,
                'status': 'indexed',
                'temp_path': None
            })
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text("🎉 Indexation terminée avec succès!")
            
            st.success(f"✅ PDF '{filename}' (ID: {file_id}) indexé avec succès dans la catégorie '{classification}'.")

            # Cleanup temporary PDF
            try:
                if os.path.exists(pdf_temp_path):
                    os.remove(pdf_temp_path)
                    print(f"Removed temporary PDF: {pdf_temp_path}")
            except Exception as cleanup_error:
                st.warning(f"Could not remove temporary PDF file '{pdf_temp_path}': {cleanup_error}")

            return updated_details

    except Exception as index_error:
        st.error(f"❌ Erreur lors de l'indexation du PDF '{filename}' (ID: {file_id}): {index_error}")
        
        updated_details.update({
            'indexed': False,
            'db_path': None,
            'status': 'error_indexing'
        })
        return updated_details

def _handle_csv_logic(file_id: str, uploaded_file, session_id: str, use_memory_limit: bool, chunk_size: int | None):
    """Handle CSV processing and return details dict."""
    try:
        csv_temp_dir = os.path.join("data", "csv_temp", session_id, file_id)
        os.makedirs(csv_temp_dir, exist_ok=True)

        original_filename = uploaded_file.name
        csv_file_path = os.path.join(csv_temp_dir, original_filename)

        # Decode file content with fallback encodings
        try:
            file_content = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                file_content = uploaded_file.getvalue().decode("latin-1")
            except UnicodeDecodeError:
                file_content = uploaded_file.getvalue().decode("iso-8859-1", errors="replace")

        with open(csv_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        # Validate CSV
        csv_processor = CSVProcessor()
        is_valid, validation_message = csv_processor.validate_csv(csv_file_path, auto_detect=True)

        if not is_valid:
            st.error(f"Le fichier CSV '{original_filename}' (ID: {file_id}) n'est pas valide: {validation_message}")
            if os.path.exists(csv_file_path):
                try: 
                    os.remove(csv_file_path)
                except Exception as rm_err: 
                    print(f"Error removing invalid CSV {csv_file_path}: {rm_err}")
            try: 
                os.rmdir(csv_temp_dir)
            except Exception as rmdir_err: 
                print(f"Error removing empty temp dir {csv_temp_dir}: {rmdir_err}")
            return None

        # Analyze CSV
        separator = csv_processor.separator
        encoding = csv_processor.encoding
        basic_analysis = csv_processor.analyze_csv(csv_file_path, auto_detect=True)
        columns = basic_analysis.get("colonnes", [])
        rows = basic_analysis.get("nombre_lignes", 0)

        file_size_mb = os.path.getsize(csv_file_path) / (1024 * 1024)
        memory_warning = ""
        if file_size_mb > 10 and not use_memory_limit:
            memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."

        figures_dir = os.path.join('data', 'figures', session_id, file_id)
        os.makedirs(figures_dir, exist_ok=True)

        csv_args = {
            "filename": original_filename,
            "file_path": csv_file_path,
            "source_file": csv_file_path,  # Add this for validation compatibility
            "separator": separator,
            "columns": columns,
            "rows": rows,
            "figures_dir": figures_dir,
            "chunk_size": chunk_size
        }

        validation_info = f"Validé avec séparateur '{separator}'"
        if memory_warning:
            validation_info += f" {memory_warning}"

        return {
            'csv_args': csv_args,
            'temp_path': csv_file_path,
            'status': 'ready',
            'columns': columns,
            'rows': rows,
            'size_mb': file_size_mb,
            'validation_info': validation_info,
            'user_notes': ''  # Initialize empty user notes
        }

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier CSV '{uploaded_file.name}' (ID: {file_id}): {str(e)}")
        if 'csv_file_path' in locals() and os.path.exists(csv_file_path):
            try: 
                os.remove(csv_file_path)
            except Exception as rm_err: 
                print(f"Error removing CSV {csv_file_path} on error: {rm_err}")
        if 'csv_temp_dir' in locals() and os.path.exists(csv_temp_dir):
            try: 
                shutil.rmtree(csv_temp_dir)
            except Exception as rmdir_err: 
                print(f"Error removing temp dir {csv_temp_dir} on error: {rmdir_err}")
        return None