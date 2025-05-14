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
        max_steps=4,  # Limite à 4 étapes pour réduire les coûts
        description="Effectue des recherches sur le web en utilisant DuckDuckGo et visite des pages web."
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
        max_steps=4,  # Limite à 4 étapes pour réduire les coûts
        description="Analyse les fichiers CSV et génère des visualisations à partir des données."
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
            max_steps=100,  # Enlever la limite d'étapes
            verbosity_level=2,
            description=(
                "Agent RAG spécialisé dans l'interrogation de documents PDF indexés dans ChromaDB. "
                "IMPORTANT: Vous avez déjà accès au contenu indexé du PDF via le seul outil disponible : RetrieverTool. "
                "N'essayez PAS d'accéder directement au fichier PDF ou d'utiliser des bibliothèques comme PyPDF2. "
                "Le document a déjà été traité et indexé dans une base de données vectorielle (ChromaDB). "
                "Vous devez simplement formuler des requêtes sémantiques pertinentes avec le RetrieverTool. "
                "\n\n"
                "INSTRUCTIONS POUR LA RECHERCHE SÉMANTIQUE EFFICACE:\n"
                "1. Pour trouver des informations dans les documents:\n"
                "   - Utilisez des termes simples et directs pour vos requêtes\n"
                "   - Essayez plusieurs variations de requêtes si nécessaire\n"
                "   - L'outil RetrieverTool essaiera automatiquement des variantes de votre requête pour améliorer les résultats\n"
                "\n"
                "2. Pour des requêtes sur des concepts spécifiques:\n"
                "   - Commencez par des requêtes directes avec les termes-clés\n"
                "   - Si nécessaire, essayez d'autres formulations\n"
                "   - Utilisez le paramètre 'additional_notes' pour ajouter du contexte à votre recherche\n"
                "\n"
                "3. Techniques de recherche améliorées:\n"
                "   - Évitez les questions complètes; préférez des expressions nominales\n"
                "   - N'incluez PAS le nom du fichier PDF ou l'extension .pdf dans vos requêtes\n"
                "   - Pour les concepts qui peuvent avoir plusieurs orthographes, essayez les différentes versions\n"
                "   - Pour les concepts complexes, décomposez en sous-requêtes plus simples\n"
                "\n"
                "4. Analyse des résultats:\n"
                "   - Examinez soigneusement les passages pour identifier les informations pertinentes\n"
                "   - Recherchez des phrases qui définissent ou expliquent les concepts demandés\n"
                "   - Synthétisez les informations de plusieurs passages si nécessaire\n"
                "\n"
                "FORMAT DE RÉPONSE REQUIS :\n"
                "Thoughts: [Vos réflexions sur la requête]\n"
                "Code:\n"
                "```python\n"
                "# Formuler une requête simple et directe pour la recherche sémantique\n"
                "search_query = \"termes clés pertinents\" \n"
                "\n"
                "# Ajouter des notes supplémentaires si pertinent\n"
                "additional_context = \"contexte supplémentaire si nécessaire\"\n"
                "\n"
                "# Utiliser le RetrieverTool pour chercher dans la base vectorielle\n"
                "results = retriever(query=search_query, additional_notes=additional_context)\n"
                "\n"
                "# Analyser les résultats\n"
                "if results and \"Retrieved documents:\" in results:\n"
                "    # Extraire les informations pertinentes\n"
                "    relevant_info = \"\"\n"
                "    # Traiter le texte pour extraire l'information demandée\n"
                "    documents = results.split(\"===== Document\")\n"
                "    \n"
                "    # Analyser chaque document pour trouver des informations pertinentes\n"
                "    for doc in documents[1:]:  # Skip the first empty element\n"
                "        # Analyser si ce document contient des informations pertinentes\n"
                "        if \"terme recherché\" in doc.lower():\n"
                "            relevant_info += doc\n"
                "    \n"
                "    if relevant_info:\n"
                "        return f\"D'après le document indexé, {relevant_info}\"\n"
                "    else:\n"
                "        # Si rien trouvé, essayer une autre requête\n"
                "        backup_results = retriever(query=\"autre formulation de recherche\", additional_notes=\"contexte supplémentaire\")\n"
                "        \n"
                "        if backup_results and \"Retrieved documents:\" in backup_results:\n"
                "            backup_docs = backup_results.split(\"===== Document\")\n"
                "            for doc in backup_docs[1:]:  # Skip the first empty element\n"
                "                if \"terme recherché\" in doc.lower():\n"
                "                    return f\"D'après le document indexé avec requête alternative, {doc}\"\n"
                "        \n"
                "        return \"Je n'ai pas trouvé d'informations spécifiques sur ce sujet dans le document indexé.\"\n"
                "else:\n"
                "    return \"Aucun document pertinent trouvé dans la base de données vectorielle.\"\n"
                "```\n"
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
        print(f"\n=== DelegateTool Debug ===")
        print(f"Agent demandé: {agent_name}")
        print(f"Requête: {user_query}")
        
        try:
            # Approche simplifiée: Utiliser les agents depuis un import explicite
            import streamlit as st
            import os
            
            if not hasattr(st, "session_state") or "agents" not in st.session_state:
                print("Erreur: session_state.agents n'est pas disponible")
                return "Erreur: Les agents ne sont pas correctement initialisés."
            
            agents = st.session_state.agents
            print(f"Agents disponibles: {list(agents.keys())}")
        
            if agent_name not in agents or agents[agent_name] is None:
                print(f"Erreur: Agent '{agent_name}' non disponible")
                return f"Erreur: Agent '{agent_name}' non disponible. Agents disponibles: {list(agents.keys())}"
            
            agent = agents[agent_name]
            print(f"Agent '{agent_name}' trouvé de type {type(agent).__name__}")
            
            # Signaler quel agent est actuellement utilisé (pour l'affichage dans le statut)
            st.session_state.current_agent = agent_name
            
            # Exécuter l'agent approprié
            if agent_name == "rag_agent":
                print("Exécution du rag_agent")
                try:
                    # Vérifier que le vectordb est configuré
                    from managed_agent.retriever_tool import get_vectordb
                    vectordb = get_vectordb()
                    print(f"⭐ RAG Agent vectordb configured: {vectordb is not None}")
                    
                    # Ajouter des logs pour le statut
                    if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                        st.session_state.status_placeholder.markdown(f"_🔍 Agent RAG en cours d'exécution pour la requête: \"{user_query}\"_")
                    
                    # Exécuter l'agent
                    result = agent.run(user_query)
                    print(f"Résultat du rag_agent (début): {result[:100]}...")
                    print(f"Résultat du rag_agent (fin): ...{result[-100:] if len(result) > 100 else result}")
                    print(f"Longueur du résultat: {len(result)}")
                    
                    # FORCER l'inclusion des sources - ajout d'un intercepteur pour vérifier si le résultat
                    # contient déjà une section de sources, et en ajouter une si ce n'est pas le cas
                    has_sources = any(marker in result for marker in ["Sources documentaires", "📚", "DEBUT_SOURCES"])
                    
                    # Si le résultat ne contient pas de section de sources, essayer d'en ajouter une
                    if not has_sources:
                        print("❗ Le résultat ne contient pas de section de sources, nous allons en ajouter une!")
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
                                    print(f"Sources ajoutées manuellement: {sources_section[:100]}...")
                        except Exception as e:
                            print(f"Erreur lors de l'ajout forcé des sources: {e}")
                            import traceback
                            print(f"TRACEBACK: {traceback.format_exc()}")
                            # Ne pas faire échouer l'exécution si l'ajout forcé échoue
                    
                    # Vérifier si le résultat contient la section des sources après les ajouts forcés
                    has_sources_section = "Sources documentaires" in result
                    print(f"Le résultat contient-il maintenant une section sources? {has_sources_section}")
                    
                    return result
                except Exception as e:
                    print(f"❌ ERROR in rag_agent execution: {e}")
                    import traceback
                    print(f"TRACEBACK: {traceback.format_exc()}")
                    return f"Erreur lors de l'exécution du rag_agent: {e}"
            elif agent_name == "search_agent":
                print("Exécution du search_agent")
                
                # Ajouter des logs pour le statut
                if hasattr(st, "session_state") and "status_placeholder" in st.session_state:
                    st.session_state.status_placeholder.markdown(f"_🔍 Agent de Recherche Web en cours d'exécution pour la requête: \"{user_query}\"_")
                
                result = agent.run(user_query)
                print(f"Résultat du search_agent: {result[:100]}...")
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
                print(f"Résultat du data_analyst: {result[:100]}...")
                return result
            else:
                return f"Agent '{agent_name}' non reconnu."
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Erreur lors de l'exécution du delegate_to_agent: {str(e)}")
            print(f"Traceback complet:\n{traceback_str}")
            return f"Erreur lors de l'exécution de l'agent {agent_name}: {str(e)}"

def initialize_manager_agent(model: OpenAIServerModel) -> CodeAgent:
    """Initializes the Manager CodeAgent."""
    return CodeAgent(
        tools=[DelegateTool()],
        model=model,
        name="manager_agent",
        max_steps=4,  # Limite à 4 étapes pour réduire les coûts
        description=(
            "Agent manager qui achemine les requêtes vers des agents spécialisés. "
            "IMPORTANT: Vous DEVEZ toujours générer un bloc de code Python valide, même en cas d'erreur. "
            "Vous n'avez PAS besoin d'accéder directement aux fichiers (PDF ou CSV). Les agents spécialisés ont déjà accès aux données. "
            "\n\n"
            "Format de réponse requis :\n"
            "Thoughts: [Vos réflexions sur la requête]\n"
            "Code:\n"
            "```python\n"
            "# Votre code Python ici\n"
            "```\n\n"
            "RÈGLES DE ROUTAGE DES REQUÊTES:\n"
            "1. Analyse fine de la requête:\n"
            "   - Identifiez s'il s'agit d'une demande de définition, d'une analyse, d'une recherche générale, etc.\n"
            "   - Déterminez si la requête concerne des documents PDF indexés, des données CSV, ou nécessite une recherche web\n"
            "\n"
            "2. Pour les définitions et concepts:\n"
            "   - Si la requête concerne un concept SPÉCIFIQUE à un PDF indexé (ex: tokenised deposits dans un document financier):\n"
            "     → Utilisez d'abord le rag_agent avec une requête précise\n"
            "   - Si le rag_agent ne trouve pas l'information OU si le concept est général:\n"
            "     → Utilisez le search_agent pour une recherche web\n"
            "\n"
            "3. Stratégie de recherche préférée:\n"
            "   - Donnez la priorité à la recherche dans les PDF indexés lorsque les termes recherchés sont mentionnés dans le contexte PDF\n"
            "   - Si un terme technique précis est demandé (comme \"tokenised deposits\"), essayez toujours PRIORITAIREMENT le rag_agent\n"
            "   - Ensuite, utilisez le search_agent comme sauvegarde si nécessaire\n"
            "\n"
            "EXEMPLES D'IMPLÉMENTATION:\n"
            "\n"
            "1. Pour interroger un document PDF indexé (définitions, explications, etc.):\n"
            "```python\n"
            "# Détection de requête de définition ou de concept technique\n"
            "definition_keywords = [\"définition\", \"definition\", \"concept\", \"what is\", \"qu'est-ce que\", \"signification\", \"meaning\"]\n"
            "is_definition_query = any(keyword in user_query.lower() for keyword in definition_keywords)\n"
            "\n"
            "# Détection de termes techniques spécifiques\n"
            "technical_terms = [\"tokenised deposits\", \"tokenized deposits\", \"cbdc\", \"stablecoin\"]\n"
            "contains_technical_term = any(term in user_query.lower() for term in technical_terms)\n"
            "\n"
            "# Si PDF disponible et requête sur définition/terme technique, essayer d'abord le RAG\n"
            "if pdf_context is not None and (is_definition_query or contains_technical_term):\n"
            "    # Essayer d'abord avec le rag_agent pour trouver dans le document indexé\n"
            "    rag_result = delegate_to_agent(\n"
            "        agent_name='rag_agent',\n"
            "        user_query=user_query,  # Transmettre la requête originale\n"
            "        context=None\n"
            "    )\n"
            "    \n"
            "    # Vérifier si le rag_agent a trouvé une réponse utile\n"
            "    not_found_indicators = [\n"
            "        \"je n'ai pas trouvé\", \n"
            "        \"aucun document pertinent\", \n"
            "        \"no relevant documents\",\n"
            "        \"aucune information\"\n"
            "    ]\n"
            "    \n"
            "    # Si le RAG n'a rien trouvé d'utile, essayer le search_agent comme fallback\n"
            "    if any(indicator in rag_result.lower() for indicator in not_found_indicators):\n"
            "        search_result = delegate_to_agent(\n"
            "            agent_name='search_agent',\n"
            "            user_query=user_query,\n"
            "            context=None\n"
            "        )\n"
            "        return f\"Le document indexé ne contenait pas l'information recherchée. Voici ce que j'ai trouvé sur le web: {search_result}\"\n"
            "    else:\n"
            "        return rag_result\n"
            "```\n"
            "\n"
            "2. Pour des recherches web:\n"
            "```python\n"
            "# Pour des informations générales ou des définitions externes\n"
            "result = delegate_to_agent(\n"
            "    agent_name='search_agent',\n"
            "    user_query=user_query,  # Utiliser la requête originale\n"
            "    context=None\n"
            ")\n"
            "return result\n"
            "```\n"
            "\n"
            "3. Pour analyser des données CSV:\n"
            "```python\n"
            "# Si des données CSV sont disponibles\n"
            "if csv_args is not None:\n"
            "    result = delegate_to_agent(\n"
            "        agent_name='data_analyst',\n"
            "        user_query=user_query,  # Utiliser la requête originale\n"
            "        context=None\n"
            "    )\n"
            "    return result\n"
            "```\n"
        )
    )

# Note: La classe RetrieverTool est maintenant définie dans managed_agent/retriever_tool.py
# La redéfinition a été supprimée pour éviter la duplication et les conflits 