import os
import streamlit as st # For potential feedback/errors
from smolagents import (
    CodeAgent, 
    ToolCallingAgent, 
    DuckDuckGoSearchTool, 
    VisitWebpageTool, 
    OpenAIServerModel,
    Tool
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.vectorstores import VectorStore
from managed_agent.retriever_tool import RetrieverTool, get_vectordb, set_vectordb
from managed_agent.vector_db_manager import get_vector_db_manager
from managed_agent.vector_config import EMBEDDING_CONFIG, get_db_path

# Define embedding model name centrally
EMBEDDING_MODEL_NAME = EMBEDDING_CONFIG["model_name"]

def initialize_search_agent(model: OpenAIServerModel) -> ToolCallingAgent:
    """Initializes the web search agent."""
    return ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        max_steps=4,
        description="Agent de recherche web utilisant DuckDuckGo et visitant des pages web pour trouver des informations pertinentes."
    )

def initialize_data_analyst_agent(model: OpenAIServerModel) -> CodeAgent:
    """Initializes the data analyst agent."""
    authorized_imports = [
        "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
        "seaborn", "io", "base64", "tempfile", "os"
    ]
    return CodeAgent(
        tools=[], # Specific tools like csv_analyzer are passed via additional_args
        model=model,
        additional_authorized_imports=authorized_imports,
        name="data_analyst",
        max_steps=4,
        description="Agent d'analyse de données CSV qui génère des statistiques et visualisations à partir de données tabulaires."
    )

def initialize_rag_agent(model: OpenAIServerModel, db_path: str) -> CodeAgent | None:
    """Initializes the RAG agent if the vector database exists."""
    try:
        print(f"RAG Agent: Attempting to load vector store from: {db_path}")
        
        # Vérification plus robuste du chemin de la base de données
        if not db_path or not isinstance(db_path, str):
            raise ValueError(f"Invalid db_path: {db_path}")
            
        # Si le chemin spécifié n'existe pas, essayer de localiser la base dans les répertoires standards
        if not os.path.exists(db_path):
            print(f"Vector DB path not found at {db_path}, looking in standard locations...")
            
            # Essayer de trouver la base dans data/output/vectordb
            standard_db_dir = os.path.join("data", "output", "vectordb")
            if os.path.exists(standard_db_dir):
                print(f"Checking in {standard_db_dir}")
                subdirs = [d for d in os.listdir(standard_db_dir) if os.path.isdir(os.path.join(standard_db_dir, d))]
                
                if subdirs:
                    found_valid_db = False
                    
                    for subdir in subdirs:
                        possible_db_path = os.path.join(standard_db_dir, subdir)
                        print(f"Examining potential DB path: {possible_db_path}")
                        
                        # Vérifier si ce répertoire contient une collection pdf_collection
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path=possible_db_path)
                            collections = client.list_collections()
                            collection_names = [c.name for c in collections]
                            
                            print(f"Found collections: {collection_names}")
                            if "pdf_collection" in collection_names:
                                print(f"Found valid vectordb at {possible_db_path} with pdf_collection")
                                db_path = possible_db_path
                                found_valid_db = True
                                break
                        except Exception as e:
                            print(f"Error checking collections in {possible_db_path}: {e}")
                            continue
                    
                    if not found_valid_db and subdirs:
                        # Si aucune collection valide n'a été trouvée mais qu'il y a des sous-répertoires,
                        # utiliser le premier comme fallback
                        db_path = os.path.join(standard_db_dir, subdirs[0])
                        print(f"No valid collection found, using first available as fallback: {db_path}")
            
            # Si toujours pas trouvé, lever l'erreur
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Vector DB path not found: {db_path}")

        # Utiliser le VectorDBManager pour gérer la connexion à la base de données
        db_manager = get_vector_db_manager(db_path)
        
        # Récupérer l'instance vectordb
        vector_store = db_manager.get_vectordb()
        
        if vector_store is None:
            # Si l'initialisation a échoué, tenter une réparation
            success, repair_msg = db_manager.repair_database()
            print(f"Database repair attempt: {repair_msg}")
            
            if success:
                vector_store = db_manager.get_vectordb()
            else:
                raise ValueError(f"Failed to initialize vector database: {repair_msg}")
        
        # Vérifier l'intégrité de la base
        integrity_ok, integrity_msg = db_manager.check_database_integrity()
        print(f"Database integrity check: {integrity_msg}")
        
        if not integrity_ok:
            print("Warning: Database integrity check failed, but continuing...")

        # 3. Initialize the RetrieverTool with detailed logging
        try:
            print("Creating RetrieverTool instance...")
            retriever_tool = RetrieverTool(vectordb=vector_store)
            
            # Vérification explicite que la base de données vectorielle est correctement définie
            if get_vectordb() is None:
                print("Warning: vectordb not properly set in RetrieverTool, setting it manually...")
                set_vectordb(vector_store)
                
            print("RetrieverTool instance created successfully")
        except Exception as tool_error:
            print(f"Error creating RetrieverTool: {tool_error}")
            raise

        # 4. Initialize the RAG Agent with specific instructions
        rag_agent = CodeAgent(
            tools=[retriever_tool],
            model=model,
            name="rag_agent",
            max_steps=100,
            verbosity_level=2,
            description=(
                "Agent RAG pour l'interrogation de documents PDF indexés dans ChromaDB. "
                "Utilisez uniquement l'outil RetrieverTool pour rechercher des informations. "
                "Conseils de recherche: "
                "- Formulez des requêtes simples avec des mots-clés pertinents "
                "- Préférez les termes nominaux aux phrases complètes "
                "- Utilisez le paramètre additional_notes au besoin pour préciser le contexte "
                "- Analysez attentivement les résultats pour synthétiser une réponse précise "
                "- Incluez toujours les sources documentaires dans votre réponse"
            )
        )
        print("RAG Agent initialized successfully with RetrieverTool.")
        return rag_agent

    except Exception as rag_init_error:
        st.error(f"Erreur lors de l'initialisation du RAG Agent : {rag_init_error}")
        print(f"RAG Agent initialization failed: {rag_init_error}")
        return None

class DelegateTool(Tool):
    name = "delegate_to_agent"
    description = "Délègue la requête à l'agent spécialisé approprié (search_agent, data_analyst, ou rag_agent)."
    inputs = {
        "agent_name": {
            "type": "string",
            "description": "Le nom de l'agent à utiliser (search_agent, data_analyst, ou rag_agent)",
        },
        "user_query": {
            "type": "string",
            "description": "La requête de l'utilisateur à traiter",
        },
        "context": {
            "type": "object",
            "description": "Le contexte contenant les agents et ressources disponibles (non utilisé)",
            "nullable": True,
            "optional": True
        }
    }
    output_type = "string"

    def forward(self, agent_name: str, user_query: str, context: dict = None) -> str:
        print(f"DelegateTool: délégation à '{agent_name}'")
        
        try:
            # Approche simplifiée: Utiliser les agents depuis un import explicite
            import streamlit as st
            import os
            
            if not hasattr(st, "session_state") or "agents" not in st.session_state:
                print("Erreur: session_state.agents n'est pas disponible")
                return "Erreur: Les agents ne sont pas correctement initialisés."
            
            agents = st.session_state.agents
            
            if agent_name not in agents or agents[agent_name] is None:
                print(f"Erreur: Agent '{agent_name}' non disponible")
                return f"Erreur: Agent '{agent_name}' non disponible. Agents disponibles: {list(agents.keys())}"
            
            agent = agents[agent_name]
            
            # Signaler quel agent est actuellement utilisé (pour l'affichage dans le statut)
            st.session_state.current_agent = agent_name
            
            # Exécuter l'agent approprié
            if agent_name == "rag_agent":
                print("Exécution du rag_agent")
                try:
                    # Vérifier que le vectordb est configuré
                    from managed_agent.retriever_tool import get_vectordb
                    vectordb = get_vectordb()
                    
                    # Ajouter des logs pour le statut
                    if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                        st.session_state.status_placeholder.markdown(f"_🔍 Agent RAG en cours d'exécution pour la requête: \"{user_query}\"_")
                    
                    # Exécuter l'agent
                    result = agent.run(user_query)
                    print(f"RAG Agent: génération d'une réponse de {len(result)} caractères")
                    
                    # FORCER l'inclusion des sources - ajout d'un intercepteur pour vérifier si le résultat
                    # contient déjà une section de sources, et en ajouter une si ce n'est pas le cas
                    has_sources = any(marker in result for marker in ["Sources documentaires", "📚", "DEBUT_SOURCES"])
                    
                    # Si le résultat ne contient pas de section de sources, essayer d'en ajouter une
                    if not has_sources:
                        print("Ajout forcé des sources documentaires")
                        try:
                            # Accéder directement à la base de données vectorielle
                            from managed_agent.retriever_tool import get_vectordb
                            vectordb = get_vectordb()
                            
                            if vectordb is not None:
                                # Effectuer une recherche pour obtenir les documents les plus pertinents
                                documents = vectordb.similarity_search(user_query, k=3)
                                
                                if documents:
                                    # Extraire les métadonnées des documents pour créer une section de sources
                                    sources_info = {}
                                    for doc in documents:
                                        metadata = getattr(doc, 'metadata', {}) or {}
                                        source = metadata.get('source', 'Unknown source')
                                        page = metadata.get('page', 'Unknown page')
                                        
                                        # Extraire le nom du fichier
                                        source_name = os.path.basename(source) if isinstance(source, str) else "Document inconnu"
                                        
                                        # Regrouper par source
                                        if source_name not in sources_info:
                                            sources_info[source_name] = set()
                                        sources_info[source_name].add(str(page))
                                    
                                    # Formater les sources pour l'affichage
                                    sources_lines = []
                                    for source_name, pages in sources_info.items():
                                        # Trier les pages numériquement
                                        page_list = sorted(pages, key=lambda x: int(x) if x.isdigit() else x)
                                        page_str = ", ".join(page_list)
                                        sources_lines.append(f"• **{source_name}** - Pages: {page_str}")
                                    
                                    # Créer la section de sources
                                    sources_section = "\n\n---\n### 📚 Sources documentaires:\n" + "\n".join(sources_lines)
                                    
                                    # Ajouter la section à la fin du résultat
                                    result = result + sources_section
                                    
                                    # Sauvegarder les sources pour affichage dans le statut
                                    st.session_state.rag_sources = [f"{source_name} (pages: {', '.join(sorted(pages, key=lambda x: int(x) if x.isdigit() else x))})" 
                                                              for source_name, pages in sources_info.items()]
                        except Exception as e:
                            print(f"Erreur lors de l'ajout forcé des sources: {e}")
                            import traceback
                            print(f"TRACEBACK: {traceback.format_exc()}")
                    
                    return result
                except Exception as e:
                    print(f"Erreur RAG Agent: {e}")
                    import traceback
                    print(f"TRACEBACK: {traceback.format_exc()}")
                    return f"Erreur lors de l'exécution du rag_agent: {e}"
            elif agent_name == "search_agent":
                print("Exécution du search_agent")
                
                # Ajouter des logs pour le statut
                if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                    st.session_state.status_placeholder.markdown(f"_🔍 Agent de Recherche Web en cours d'exécution pour la requête: \"{user_query}\"_")
                
                result = agent.run(user_query)
                print(f"Search Agent: réponse générée")
                return result
            elif agent_name == "data_analyst":
                print("Exécution du data_analyst")
                
                # Ajouter des logs pour le statut
                if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                    st.session_state.status_placeholder.markdown(f"_📊 Agent d'Analyse de Données en cours d'exécution pour la requête: \"{user_query}\"_")
                
                csv_args = None
                # Récupérer les csv_args depuis st.session_state si disponibles
                for file_id, details in st.session_state.processed_files.items():
                    if details.get('type') == 'csv' and details.get('status') == 'ready':
                        csv_args = details.get('csv_args')
                        
                        # Ajouter le nom du fichier CSV au statut
                        if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                            filename = details.get('filename', 'CSV sans nom')
                            st.session_state.status_placeholder.markdown(f"_📊 Analyse du fichier CSV: {filename}_")
                        break
                
                if csv_args:
                    result = agent.run(user_query, additional_args={"csv_analyzer": csv_args})
                else:
                    result = agent.run(user_query)
                print(f"Data Analyst: réponse générée")
                return result
            else:
                return f"Agent '{agent_name}' non reconnu."
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Erreur DelegateTool: {str(e)}")
            print(f"Traceback: {traceback_str}")
            return f"Erreur lors de l'exécution de l'agent {agent_name}: {str(e)}"

def initialize_manager_agent(model: OpenAIServerModel) -> CodeAgent:
    """Initializes the Manager CodeAgent."""
    return CodeAgent(
        tools=[DelegateTool()],
        model=model,
        name="manager_agent",
        max_steps=4,
        description=(
            "Agent manager qui route les requêtes vers des agents spécialisés. "
            "Utilisez les agents disponibles (search_agent, data_analyst, rag_agent) selon le type de requête. "
            "Priorité aux documents PDF indexés pour les questions techniques spécifiques. "
            "Utilisez le search_agent pour des infos générales ou comme fallback. "
            "Utilisez le data_analyst pour l'analyse CSV. "
            "Générez toujours un code Python utilisant la fonction delegate_to_agent()."
        )
    )

# Note: La classe RetrieverTool est maintenant définie dans managed_agent/retriever_tool.py
# La redéfinition a été supprimée pour éviter la duplication et les conflits 