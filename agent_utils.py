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
from managed_agent.retriever_tool import RetrieverTool

# Define embedding model name centrally
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
            
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector DB path not found: {db_path}")
            
        # Vérifier si le répertoire contient des fichiers ChromaDB
        chroma_files = [f for f in os.listdir(db_path) if f.endswith('.sqlite3') or f == 'chroma.sqlite3']
        if not chroma_files:
            # Vérifier s'il y a au moins quelques fichiers attendus
            all_files = os.listdir(db_path)
            print(f"Files in DB directory: {all_files}")
            
            if len(all_files) < 2:  # Généralement, il devrait y avoir plusieurs fichiers
                raise FileNotFoundError(f"Vector DB directory exists but appears empty: {db_path}")
            else:
                print(f"Warning: No direct SQLite files found in {db_path}, but directory contains {len(all_files)} files")

        # 1. Initialize the embedding model using the LangChain wrapper
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # 2. Initialize the Chroma vector store with the wrapped embedding function
        print(f"Initializing Chroma vector store with persist_directory={db_path}")
        try:
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embedding_function
            )
            
            # Test simple de connexion
            try:
                test_results = vector_store.similarity_search("test connection", k=1)
                print(f"Database connection test successful: found {len(test_results)} results")
            except Exception as test_error:
                print(f"Database connection test error: {test_error}")
                print("Attempting to fix the connection...")
                
                # Si le test a échoué, essayons de recréer la connection avec un autre client
                import chromadb
                client = chromadb.PersistentClient(path=db_path)
                try:
                    collections = client.list_collections()
                    print(f"Found {len(collections)} collections in ChromaDB")
                    if collections:
                        collection_name = collections[0].name
                        print(f"Using collection: {collection_name}")
                        # Recréer le Chroma store avec ce client explicite
                        vector_store = Chroma(
                            client=client,
                            collection_name=collection_name,
                            embedding_function=embedding_function
                        )
                        print("Successfully recreated Chroma connection")
                    else:
                        raise ValueError("No collections found in the database")
                except Exception as collection_error:
                    print(f"Error accessing collections: {collection_error}")
                    raise
        except Exception as chroma_error:
            print(f"Error initializing Chroma: {chroma_error}")
            # Essayer une approche alternative si la première échoue
            try:
                print("Trying alternative Chroma initialization...")
                import chromadb
                client = chromadb.PersistentClient(path=db_path)
                
                # Lister les collections disponibles
                collections = client.list_collections()
                if not collections:
                    raise ValueError(f"No collections found in database at {db_path}")
                
                collection_name = collections[0].name
                print(f"Using collection: {collection_name}")
                
                vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embedding_function
                )
                print("Alternative Chroma initialization successful")
            except Exception as alt_error:
                print(f"Alternative initialization failed: {alt_error}")
                raise

        # 3. Initialize the RetrieverTool with detailed logging
        retriever_tool = RetrieverTool(vectordb=vector_store)

        # 4. Initialize the RAG Agent with specific instructions
        rag_agent = CodeAgent(
            tools=[retriever_tool],
            model=model,
            name="rag_agent",
            max_steps=4,  # Limite à 4 étapes pour réduire les coûts
            verbosity_level=2,
            description=(
                "Agent RAG spécialisé dans l'interrogation de documents PDF indexés dans ChromaDB. "
                "IMPORTANT: Vous avez déjà accès au contenu indexé du PDF via le seul outil disponible : RetrieverTool. "
                "N'essayez PAS d'accéder directement au fichier PDF ou d'utiliser des bibliothèques comme PyPDF2. "
                "Le document a déjà été traité et indexé dans une base de données vectorielle (ChromaDB). "
                "Vous devez simplement formuler des requêtes sémantiques pertinentes avec le RetrieverTool. "
                "\n\n"
                "INSTRUCTIONS POUR LA RECHERCHE SÉMANTIQUE EFFICACE:\n"
                "1. Pour trouver une définition ou un concept:\n"
                "   - Utilisez des termes simples et directs comme 'tokenised deposits definition' ou 'tokenized deposits concept'\n"
                "   - Essayez plusieurs variations de requêtes si la première ne donne pas de résultats pertinents\n"
                "   - L'outil RetrieverTool essaiera automatiquement des variantes de votre requête pour améliorer les résultats\n"
                "\n"
                "2. Pour des requêtes sur des termes spécifiques comme 'tokenised deposits':\n"
                "   - Commencez par une requête directe : 'tokenised deposits definition'\n"
                "   - Si nécessaire, essayez d'autres formulations : 'what are tokenised deposits' ou 'tokenised deposits explanation'\n"
                "   - Utilisez le paramètre 'additional_notes' pour ajouter du contexte: additional_notes='looking for a formal definition from BIS or financial regulations'\n"
                "\n"
                "3. Techniques de recherche améliorées:\n"
                "   - Évitez les questions complètes; préférez des expressions nominales\n"
                "   - N'incluez PAS le nom du fichier PDF ou l'extension .pdf dans vos requêtes\n"
                "   - Si vous recherchez un concept qui peut avoir plusieurs orthographes, essayez les deux versions (ex: 'tokenized' et 'tokenised')\n"
                "   - Pour les concepts complexes, décomposez en sous-requêtes plus simples\n"
                "\n"
                "4. Analyse des résultats:\n"
                "   - Examinez soigneusement les résultats pour identifier les passages contenant des définitions formelles\n"
                "   - Recherchez des phrases comme 'X are defined as', 'X refer to', 'X means'\n"
                "   - Parfois, les définitions sont introduites par des phrases comme 'In this paper, X refers to...'\n"
                "\n"
                "FORMAT DE RÉPONSE REQUIS :\n"
                "Thoughts: [Vos réflexions sur la requête]\n"
                "Code:\n"
                "```python\n"
                "# Formuler une requête simple et directe pour la recherche sémantique\n"
                "search_query = \"tokenised deposits definition\" \n"
                "\n"
                "# Ajouter des notes supplémentaires si pertinent\n"
                "additional_context = \"looking for formal definition in financial documents\"\n"
                "\n"
                "# Utiliser le RetrieverTool pour chercher dans la base vectorielle\n"
                "results = retriever(query=search_query, additional_notes=additional_context)\n"
                "\n"
                "# Analyser les résultats\n"
                "if results and \"Retrieved documents:\" in results:\n"
                "    # Extraire les informations pertinentes\n"
                "    relevant_info = \"\"\n"
                "    # Traiter le texte pour extraire la définition ou l'information demandée\n"
                "    documents = results.split(\"===== Document\")\n"
                "    \n"
                "    # Chercher d'abord les définitions explicites\n"
                "    definition_markers = [\"are defined as\", \"refers to\", \"is defined as\", \"definition\", \"concept of\", \"means\"]\n"
                "    definition_found = False\n"
                "    \n"
                "    for doc in documents[1:]:  # Skip the first empty element\n"
                "        doc_lower = doc.lower()\n"
                "        if \"tokenised deposits\" in doc_lower or \"tokenized deposits\" in doc_lower:\n"
                "            # Chercher en priorité des marqueurs de définition\n"
                "            for marker in definition_markers:\n"
                "                if marker in doc_lower:\n"
                "                    relevant_info += doc\n"
                "                    definition_found = True\n"
                "                    break\n"
                "            \n"
                "            # Si aucun marqueur trouvé mais contient quand même les termes cherchés\n"
                "            if not definition_found:\n"
                "                relevant_info += doc\n"
                "    \n"
                "    if relevant_info:\n"
                "        return f\"D'après le document indexé, {relevant_info}\"\n"
                "    else:\n"
                "        # Si rien trouvé, essayer une autre requête\n"
                "        backup_results = retriever(query=\"what are tokenised deposits\", additional_notes=\"looking for explanation or description\")\n"
                "        \n"
                "        if backup_results and \"Retrieved documents:\" in backup_results:\n"
                "            backup_docs = backup_results.split(\"===== Document\")\n"
                "            for doc in backup_docs[1:]:  # Skip the first empty element\n"
                "                if \"tokenised deposits\" in doc.lower() or \"tokenized deposits\" in doc.lower():\n"
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
            
            # Exécuter l'agent approprié
            if agent_name == "rag_agent":
                print("Exécution du rag_agent")
                result = agent.run(user_query)
                print(f"Résultat du rag_agent: {result[:100]}...")
                return result
            elif agent_name == "search_agent":
                print("Exécution du search_agent")
                result = agent.run(user_query)
                print(f"Résultat du search_agent: {result[:100]}...")
                return result
            elif agent_name == "data_analyst":
                print("Exécution du data_analyst")
                csv_args = None
                # Récupérer les csv_args depuis st.session_state si disponibles
                for file_id, details in st.session_state.processed_files.items():
                    if details.get('type') == 'csv' and details.get('status') == 'ready':
                        csv_args = details.get('csv_args')
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