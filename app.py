import streamlit as st
import os
import shutil
import uuid
import tempfile
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel
from managed_agent.csv_analyzer_tool import CSVAnalyzerTool
from utils.pipeline_indexation.csv_processor import CSVProcessor

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

    # Initialisation de la session si nécessaire.
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Dossier spécifique pour les figures de cette session.
    figures_dir = os.path.join('./figures', st.session_state.session_id)
    
    # Récupération de la clé API depuis les variables d'environnement.
    api_key_from_env = os.environ.get("OPENAI_API_KEY")
    
    # Saisie de la clé API seulement si elle n'est pas déjà définie.
    if not api_key_from_env:
        api_key = st.text_input("Clé API OpenAI", type="password")
    else:
        st.success("Clé API OpenAI trouvée dans les variables d'environnement")
        api_key = api_key_from_env
    
    # Upload d'un fichier CSV (optionnel).
    uploaded_file = st.file_uploader("Déposer un fichier CSV (optionnel)", type=['csv'])
    
    # Option pour limiter l'utilisation de la mémoire.
    use_memory_limit = st.checkbox("Limiter l'utilisation de la mémoire (recommandé pour les grands fichiers)", value=True)
    chunk_size = 100000 if use_memory_limit else None
    
    # Permettre à l'utilisateur d'ajouter ses propres notes sur le fichier
    user_notes = ""
    if uploaded_file:
        user_notes = st.text_area("Notes additionnelles sur le fichier", 
                                  placeholder="Ajoutez ici vos observations, questions spécifiques ou contexte sur le fichier CSV...",
                                  height=150)
    
    # Saisie de la requête utilisateur.
    user_query = st.text_input("Requête à envoyer à l'agent")
    
    if st.button("Exécuter"):
        # Vérification des entrées.
        if not api_key:
            st.error("Veuillez entrer une clé API valide.")
            return
        if not user_query:
            st.error("Veuillez entrer une requête.")
            return
        
        # Nettoyer les ressources précédentes.
        cleanup_resources(figures_dir)
        
        # Définition de la clé API dans l'environnement.
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
        
        # Préparation des paramètres liés au fichier CSV s'il y a lieu.
        csv_args = None
        if uploaded_file is not None:
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
                file_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
                with open(csv_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                
                # Utiliser CSVProcessor pour valider et analyser le fichier
                csv_processor = CSVProcessor()
                is_valid, validation_message = csv_processor.validate_csv(csv_file_path, auto_detect=True)
                
                if not is_valid:
                    st.error(f"Le fichier CSV n'est pas valide: {validation_message}")
                    return
                
                # Récupérer les informations détectées
                separator = csv_processor.separator
                encoding = csv_processor.encoding
                
                # Analyse basique pour obtenir des informations sur le fichier
                basic_analysis = csv_processor.analyze_csv(csv_file_path, auto_detect=True)
                columns = basic_analysis.get("colonnes", [])
                rows = basic_analysis.get("nombre_lignes", 0)
                
                # Estimer la taille du fichier pour recommandations de mémoire
                file_size_mb = len(file_content) / (1024 * 1024)
                memory_warning = ""
                if file_size_mb > 10 and not use_memory_limit:
                    memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."
                
                # Préparer les notes pour le data_analyst
                data_analyst_notes = f"""
# Guide d'analyse de données
- Fichier: {original_filename}
- Chemin: {csv_file_path}
- Séparateur: '{separator}'
- Encodage: {encoding}
- Dossier pour figures: '{figures_dir}'
- Taille du fichier: {file_size_mb:.2f} MB
- Nombre de lignes: {rows}
- Nombre de colonnes: {len(columns)}

# Colonnes détectées:
{', '.join(columns)}

# Notes de l'utilisateur:
{user_notes if user_notes else "Aucune note spécifique fournie par l'utilisateur."}

# Étapes recommandées:
1. Validation du fichier CSV via csv_analyzer (déjà effectuée: {validation_message})
2. Exploration des données avec df.info(), df.describe() et vérification des valeurs manquantes
3. Création de visualisations adaptées au type de données
4. Enregistrer toutes les figures dans le dossier '{figures_dir}'
                """
                
                csv_args = {
                    "source_file": csv_file_path, 
                    "separator": separator, 
                    "additional_notes": data_analyst_notes,
                    "figures_dir": figures_dir,
                    "chunk_size": chunk_size
                }
                
                info_message = f"Fichier CSV validé avec séparateur '{separator}' et encodage '{encoding}'. L'agent Data Analyst sera utilisé."
                if memory_warning:
                    info_message += f" {memory_warning}"
                st.info(info_message)
            
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier CSV: {str(e)}")
                return
        else:
            st.info("Aucun fichier CSV détecté. L'agent de recherche sera utilisé.")
        
        # Utilisation de la fonction de routage pour déléguer la requête.
        with st.spinner("L'agent traite votre requête..."):
            try:
                result = route_request(user_query, csv_args, search_agent, data_analyst)
            except Exception as e:
                st.error(f"Erreur lors du traitement de la requête: {str(e)}")
                return
        
        st.subheader("Résultat de l'agent")
        st.markdown(result, unsafe_allow_html=True)
        
        # Afficher les figures générées dans le dossier des figures, si présentes.
        if os.path.exists(figures_dir) and os.listdir(figures_dir):
            st.subheader("Figures générées")
            for fig_file in os.listdir(figures_dir):
                if fig_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(os.path.join(figures_dir, fig_file), caption=fig_file)
        
        # Conseil complémentaire pour l'analyse de données.
        if csv_args is not None:
            st.info("💡 Conseil: Vous pouvez demander à l'agent d'effectuer des analyses plus spécifiques, comme des corrélations ou des statistiques détaillées sur vos données CSV.")

if __name__ == "__main__":
    main()