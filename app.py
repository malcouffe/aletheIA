import streamlit as st
import os
import shutil
import uuid
import torch # Ensure torch is imported
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel
from utils.pipeline_indexation.csv_processor import CSVProcessor
from utils.pipeline_indexation.pdf_processor import analyser_pdf  # Import the PDF processor

# Workaround for Streamlit/Torch watcher warning
# See: https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/5
torch.classes.__path__ = []

def route_request(query, csv_args, search_agent, data_analyst):
    """
    Fonction de routage qui délègue la requête à l'agent spécialisé.
    Si csv_args est fourni (indiquant la présence d'un fichier CSV), la requête est déléguée à data_analyst.
    Sinon, la requête est envoyée à search_agent.
    """
    if csv_args is not None:
        # Définir le prompt selon la présence ou non des additional_notes
        additional_notes = csv_args.get('additional_notes', '').strip()
        
        # Message de base indiquant l'expertise en data-analysis
        expertise_message = (
            "Vous êtes un expert en data-analysis. "
            "Votre tâche est d'analyser le fichier CSV fourni afin de répondre à la question posée. "
        )
        
        # Construire le prompt en incluant les notes additionnelles
        prompt = (
            f"{expertise_message}\n"
            f"Analyse du fichier CSV: {query}\n\n"
            f"Notes additionnelles et contexte: {additional_notes}"
        )

        # Préparer les arguments pour l'outil csv_analyzer.
        csv_analyzer_args = {
            "source_file": csv_args["source_file"],
            "separator": csv_args["separator"],
            "figures_dir": csv_args["figures_dir"],
            "chunk_size": csv_args["chunk_size"]
        }
        
        return data_analyst.run(prompt, additional_args={"csv_analyzer": csv_analyzer_args})
    else:
        # Délégation à l'agent de recherche pour traiter la requête générale.
        return search_agent.run(query)

def cleanup_resources(figures_dir):
    """
    Nettoie les ressources utilisées par les agents.
    
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

def main():
    st.title("Agent Web avec Streamlit")
    st.write("Entrez votre clé API OpenAI et votre requête pour interroger l'agent.")

    # --- Configuration de la Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        # Récupération de la clé API depuis les variables d'environnement.
        api_key_from_env = os.environ.get("OPENAI_API_KEY")
        mistral_api_key_from_env = os.environ.get("MISTRAL_API_KEY") # Get Mistral key

        # Saisie de la clé API OpenAI seulement si elle n'est pas déjà définie.
        if not api_key_from_env:
            api_key = st.text_input("Clé API OpenAI", type="password")
        else:
            st.success("Clé API OpenAI trouvée")
            api_key = api_key_from_env
            
        # Saisie de la clé API Mistral si nécessaire pour le PDF
        if not mistral_api_key_from_env:
             mistral_api_key = st.text_input("Clé API Mistral (pour PDF)", type="password")
        else:
            st.success("Clé API Mistral trouvée")
            mistral_api_key = mistral_api_key_from_env

        st.subheader("Chargement de Fichier")
        # Upload d'un fichier CSV ou PDF (optionnel).
        uploaded_file = st.file_uploader("Déposer un fichier (CSV ou PDF)", type=['csv', 'pdf'])

        # Options spécifiques au fichier (affichées si un fichier est chargé)
        user_notes = ""
        use_memory_limit = True # Default value for CSV
        if uploaded_file:
            file_type = uploaded_file.type
            if file_type == 'text/csv':
                # Option pour limiter l'utilisation de la mémoire (CSV).
                use_memory_limit = st.checkbox("Limiter l'utilisation de la mémoire (CSV)", value=True)
            # Permettre à l'utilisateur d'ajouter ses propres notes sur le fichier
            user_notes = st.text_area("Notes additionnelles sur le fichier",
                                      placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur le fichier...",
                                      height=150)
    # -------------------------------------

    # --- Interface Principale ---
    # Initialisation de la session si nécessaire.
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pdf_processed = False # Track PDF processing state
        st.session_state.processed_pdf_path = None
        st.session_state.pdf_summary = None

    # Dossier spécifique pour les figures de cette session.
    figures_dir = os.path.join('./figures', st.session_state.session_id)

    chunk_size = 100000 if use_memory_limit else None # Défini après la checkbox

    # Saisie de la requête utilisateur.
    user_query = st.text_input("Requête à envoyer à l'agent")

    if st.button("Exécuter"):
        # Vérification des entrées.
        if not api_key:
            st.error("Veuillez entrer une clé API OpenAI valide.")
            return
        # Check Mistral key only if a PDF is uploaded or was processed previously
        if (uploaded_file and uploaded_file.type == 'application/pdf') or st.session_state.get('processed_pdf_path'):
            if not mistral_api_key:
                 st.error("Veuillez entrer une clé API Mistral valide pour le traitement PDF.")
                 return
            os.environ["MISTRAL_API_KEY"] = mistral_api_key # Set Mistral key in env

        if not user_query and not (uploaded_file and uploaded_file.type == 'application/pdf'): # Allow execution just for PDF processing
             st.error("Veuillez entrer une requête.")
             return

        # Nettoyer les ressources précédentes (figures).
        cleanup_resources(figures_dir)

        # Définition de la clé API OpenAI dans l'environnement.
        os.environ["OPENAI_API_KEY"] = api_key

        # Création du modèle OpenAIServerModel.
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=api_key,
        )

        # Création de l'agent de recherche (search_agent).
        search_agent = ToolCallingAgent(
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=model,
            name="search_agent",
            description="Effectue des recherches sur le web en utilisant DuckDuckGo et visite des pages web."
        )

        # Création de l'agent d'analyse des données (data_analyst).
        authorized_imports = [
            "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
            "seaborn", "io", "base64", "tempfile", "os"
        ]
        data_analyst = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=authorized_imports,
            name="data_analyst",
            description="Analyse les fichiers CSV et génère des visualisations à partir des données."
        )

        # Liste des agents gérés.
        managed_agents = [search_agent, data_analyst]

        # Préparation des paramètres liés aux fichiers.
        csv_args = None
        pdf_processed_this_run = False # Track if PDF was processed in this specific run

        if uploaded_file is not None:
            file_type = uploaded_file.type
            
            # --- Traitement PDF ---
            if file_type == 'application/pdf':
                 if not mistral_api_key: # Double check Mistral key
                     st.error("Clé API Mistral requise pour traiter le PDF.")
                     return

                 try:
                    # Créer un répertoire temporaire pour le PDF
                    pdf_temp_dir = os.path.join("data", "pdf_temp", st.session_state.session_id)
                    os.makedirs(pdf_temp_dir, exist_ok=True)

                    # Sauvegarder le PDF uploadé
                    pdf_path = os.path.join(pdf_temp_dir, uploaded_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Appeler la fonction d'analyse PDF
                    with st.spinner("Analyse du PDF en cours (cela peut prendre du temps)..."):
                        pdf_summary = analyser_pdf(pdf_path)
                        st.session_state.pdf_processed = True
                        st.session_state.processed_pdf_path = pdf_path # Store path for context
                        st.session_state.pdf_summary = pdf_summary # Store summary
                        pdf_processed_this_run = True # Mark PDF as processed in this run
                        st.success(f"PDF '{uploaded_file.name}' analysé et indexé avec succès.")
                        st.json(pdf_summary) # Display summary info

                    # Nettoyer le fichier PDF temporaire après analyse (garder le dossier pour output)
                    # os.remove(pdf_path) # Keep the original for now if needed later

                 except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier PDF: {str(e)}")
                    # Nettoyer le dossier temporaire en cas d'erreur
                    if os.path.exists(pdf_temp_dir):
                        shutil.rmtree(pdf_temp_dir)
                    return # Stop execution on PDF processing error

            # --- Traitement CSV ---
            elif file_type == 'text/csv':
                try:
                    # Créer un répertoire pour stocker les fichiers CSV
                    csv_dir = os.path.join("data", "csv_files")
                    os.makedirs(csv_dir, exist_ok=True)

                    # Générer un nom de fichier unique basé sur le nom original
                    original_filename = uploaded_file.name
                    base_name = os.path.splitext(original_filename)[0]
                    unique_filename = f"{base_name}_{str(uuid.uuid4())[:8]}.csv"
                    csv_file_path = os.path.join(csv_dir, unique_filename)

                    # Lire et sauvegarder le contenu du fichier
                    # Try common encodings if utf-8 fails
                    try:
                        file_content = uploaded_file.getvalue().decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            file_content = uploaded_file.getvalue().decode("latin-1")
                        except UnicodeDecodeError:
                             file_content = uploaded_file.getvalue().decode("iso-8859-1", errors="replace")

                    with open(csv_file_path, "w", encoding="utf-8") as f: # Save as utf-8
                        f.write(file_content)

                    # Utiliser CSVProcessor pour valider et analyser le fichier
                    csv_processor = CSVProcessor()
                    is_valid, validation_message = csv_processor.validate_csv(csv_file_path, auto_detect=True)

                    if not is_valid:
                        st.error(f"Le fichier CSV n'est pas valide: {validation_message}")
                        return

                    # Récupérer les informations détectées
                    separator = csv_processor.separator
                    encoding = csv_processor.encoding # Keep detected encoding info

                    # Analyse basique pour obtenir des informations sur le fichier
                    basic_analysis = csv_processor.analyze_csv(csv_file_path, auto_detect=True)
                    columns = basic_analysis.get("colonnes", [])
                    rows = basic_analysis.get("nombre_lignes", 0)

                    # Estimer la taille du fichier pour recommandations de mémoire
                    file_size_mb = os.path.getsize(csv_file_path) / (1024 * 1024)
                    memory_warning = ""
                    if file_size_mb > 10 and not use_memory_limit:
                        memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."

                    # Préparer les notes pour le data_analyst
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
                    return
            else:
                 st.warning(f"Type de fichier non supporté: {file_type}")
                 return # Stop if unsupported file type uploaded

        # --- Exécution de la requête ---
        # Only run route_request if there's a query OR if a CSV was just uploaded
        # If only a PDF was uploaded, we skip routing for now.
        if user_query or csv_args:
            # Check if a PDF was processed in a previous run
            pdf_context_message = ""
            if st.session_state.get('pdf_processed') and st.session_state.get('processed_pdf_path'):
                 pdf_context_message = f"\n\nContext: Un fichier PDF a été préalablement analysé ({os.path.basename(st.session_state.processed_pdf_path)}). Utilisez les informations de ce PDF si pertinent pour la requête."
                 # TODO: Implement actual RAG query mechanism here instead of just adding context message

            # Add PDF context to the query if applicable
            full_query = user_query + pdf_context_message

            # Utilisation de la fonction de routage pour déléguer la requête.
            with st.spinner("L'agent traite votre requête..."):
                try:
                    # Pass the modified query with PDF context
                    result = route_request(full_query, csv_args, search_agent, data_analyst) 
                except Exception as e:
                    st.error(f"Erreur lors du traitement de la requête: {str(e)}")
                    return # Stop execution on routing/agent error

            st.subheader("Résultat de l'agent")
            st.markdown(result, unsafe_allow_html=True)

            # Afficher les figures générées dans le dossier des figures, si présentes (CSV analysis).
            if csv_args and os.path.exists(figures_dir) and os.listdir(figures_dir):
                st.subheader("Figures générées (Analyse CSV)")
                for fig_file in os.listdir(figures_dir):
                    if fig_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        st.image(os.path.join(figures_dir, fig_file), caption=fig_file)

            # Conseil complémentaire pour l'analyse de données CSV.
            if csv_args is not None:
                st.info("💡 Conseil: Vous pouvez demander à l'agent d'effectuer des analyses plus spécifiques, comme des corrélations ou des statistiques détaillées sur vos données CSV.")
        elif pdf_processed_this_run:
             st.info("Le fichier PDF a été traité et indexé. Vous pouvez maintenant poser des questions à son sujet.")
        else:
             # This case should not happen if validation is correct, but as a fallback:
             st.warning("Aucune action effectuée. Veuillez fournir une requête ou un fichier supporté.")


if __name__ == "__main__":
    main()