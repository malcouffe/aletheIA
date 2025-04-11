import streamlit as st
import os
import tempfile
import shutil
import uuid
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel
from managed_agent.csv_analyzer_tool import CSVAnalyzerTool

def route_request(query, csv_args, search_agent, data_analyst):
    """
    Fonction de routage qui délègue la requête à l'agent spécialisé.
    Si csv_args est fourni (indiquant la présence d'un fichier CSV), la requête est déléguée à data_analyst.
    Sinon, la requête est envoyée à search_agent.
    """
    if csv_args is not None:
        # Délégation à l'agent d'analyse de données pour traitement CSV
        return data_analyst.run("Analyse du fichier CSV: " + query, additional_args=csv_args)
    else:
        # Délégation à l'agent de recherche pour traiter la requête générale
        return search_agent.run(query)

def cleanup_resources(figures_dir):
    """
    Nettoie les ressources utilisées par les agents.
    
    Args:
        figures_dir: Chemin vers le dossier contenant les figures générées
    """
    # Nettoyage du dossier de figures s'il existe
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

    # Initialisation de la session si nécessaire
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Dossier spécifique pour les figures de cette session
    figures_dir = os.path.join('./figures', st.session_state.session_id)
    
    # Saisie de la clé API (en mode mot de passe)
    api_key = st.text_input("Clé API OpenAI", type="password")
    
    # Upload d'un fichier CSV (optionnel)
    uploaded_file = st.file_uploader("Déposer un fichier CSV (optionnel)", type=['csv'])
    
    # Option pour limiter l'utilisation de la mémoire
    use_memory_limit = st.checkbox("Limiter l'utilisation de la mémoire (recommandé pour les grands fichiers)", value=True)
    chunk_size = 100000 if use_memory_limit else None
    
    # Saisie de la requête utilisateur
    user_query = st.text_input("Requête à envoyer à l'agent")
    
    if st.button("Exécuter"):
        # Vérification des entrées
        if not api_key:
            st.error("Veuillez entrer une clé API valide.")
            return
        if not user_query:
            st.error("Veuillez entrer une requête.")
            return
        
        # Nettoyer les ressources précédentes
        cleanup_resources(figures_dir)
        
        # Définition de la clé API dans l'environnement
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Création du modèle OpenAIServerModel
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=api_key,
        )
        
        # Création de l'agent de recherche (search_agent)
        search_agent = ToolCallingAgent(
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=model,
            name="search_agent",
            description="Effectue des recherches sur le web en utilisant DuckDuckGo et visite des pages web."
        )
        
        # Création de l'agent d'analyse des données (data_analyst)
        authorized_imports = [
            "pandas", "numpy", "matplotlib", "matplotlib.pyplot", 
            "seaborn", "io", "base64", "tempfile", "os"
        ]
        data_analyst = CodeAgent(
            tools=[CSVAnalyzerTool()],
            model=model,
            additional_authorized_imports=authorized_imports,
            name="data_analyst",
            description="Analyse les fichiers CSV et génère des visualisations à partir des données."
        )
        
        # Liste des agents gérés
        managed_agents = [search_agent, data_analyst]
        
        # Préparation des paramètres liés au fichier CSV s'il y a lieu
        csv_args = None
        if uploaded_file is not None:
            try:
                # Lire le contenu du fichier uploadé
                file_content = uploaded_file.getvalue().decode("utf-8")
                # Déterminer le séparateur via un heuristique simple
                first_line = file_content.split('\n')[0]
                separator = ',' if ',' in first_line else ';'
                # Sauvegarder le contenu dans un fichier temporaire
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                csv_file_path = temp_file.name
                temp_file.write(file_content.encode("utf-8"))
                temp_file.close()
                
                # Estimer la taille du fichier pour recommandations de mémoire
                file_size_mb = len(file_content) / (1024 * 1024)
                memory_warning = ""
                if file_size_mb > 10 and not use_memory_limit:
                    memory_warning = "⚠️ Le fichier est relativement volumineux. L'option de limitation de mémoire est recommandée."
                
                # Préparer les notes pour le data_analyst
                data_analyst_notes = f"""
# Guide d'analyse de données
- Chemin: {csv_file_path}
- Séparateur: '{separator}'
- Dossier pour figures: '{figures_dir}'
- Taille du fichier: {file_size_mb:.2f} MB

# Étapes recommandées :
1. Validation du fichier CSV via csv_analyzer.
2. Exploration des données avec df.info(), df.describe() et vérification des valeurs manquantes.
3. Création de visualisations (histplots, boxplots, scatter plots) avec seaborn et matplotlib.
4. Enregistrer toutes les figures dans le dossier '{figures_dir}'.
                """
                csv_args = {
                    "source_file": csv_file_path, 
                    "separator": separator, 
                    "additional_notes": data_analyst_notes,
                    "figures_dir": figures_dir,
                    "chunk_size": chunk_size
                }
                
                info_message = f"Fichier CSV détecté avec séparateur '{separator}'. L'agent Data Analyst sera utilisé."
                if memory_warning:
                    info_message += f" {memory_warning}"
                st.info(info_message)
            
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier CSV: {str(e)}")
                return
        else:
            st.info("Aucun fichier CSV détecté. L'agent de recherche sera utilisé.")
        
        # Utilisation de la fonction de routage pour déléguer la requête
        with st.spinner("L'agent traite votre requête..."):
            try:
                result = route_request(user_query, csv_args, search_agent, data_analyst)
            except Exception as e:
                st.error(f"Erreur lors du traitement de la requête: {str(e)}")
                return
        
        st.subheader("Résultat de l'agent")
        st.markdown(result, unsafe_allow_html=True)
        
        # Afficher les figures générées dans le dossier des figures, si présentes
        if os.path.exists(figures_dir) and os.listdir(figures_dir):
            st.subheader("Figures générées")
            for fig_file in os.listdir(figures_dir):
                if fig_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(os.path.join(figures_dir, fig_file), caption=fig_file)
        
        # Nettoyage du fichier temporaire si uploadé
        if uploaded_file is not None and csv_args is not None:
            try:
                os.unlink(csv_args["source_file"])
            except Exception as e:
                st.error(f"Erreur lors de la suppression du fichier temporaire: {e}")
        
        # Conseil complémentaire pour l'analyse de données
        if csv_args is not None:
            st.info("💡 Conseil: Vous pouvez demander à l'agent d'effectuer des analyses plus spécifiques, comme des corrélations ou des statistiques détaillées sur vos données CSV.")

if __name__ == "__main__":
    main()
