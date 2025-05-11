from smolagents import Tool
from langchain_core.vectorstores import VectorStore
import os


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "additional_notes": {
            "type": "string",
            "description": "Optional additional notes or context to refine the search query.",
            "optional": True,
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        print(f"RetrieverTool initialized with vector store at: {vectordb._persist_directory}")

    def forward(self, query: str, additional_notes: str | None = None) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        print(f"\n=== RetrieverTool Search ===")
        print(f"Query: {query}")
        print(f"Additional notes: {additional_notes}")

        # Vérifier que la base de données est accessible
        try:
            # Test simple de connexion à la base
            persist_directory = getattr(self.vectordb, '_persist_directory', None)
            if persist_directory:
                print(f"Checking database at: {persist_directory}")
                if not os.path.exists(persist_directory):
                    raise ValueError(f"Database directory does not exist: {persist_directory}")
            else:
                # Si _persist_directory n'est pas disponible, essayons une autre approche
                print("Database path attribute not found, attempting query directly")
                self.vectordb.similarity_search("test", k=1)
                print("Database connection test successful")
        except Exception as db_error:
            print(f"Database connection error: {db_error}")
            # Tenter de recréer la connexion à la base de données
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                # Récupérer le chemin de la base et recréer la connexion
                print("Attempting to reconnect to database...")
                if persist_directory:
                    print(f"Reconnecting to: {persist_directory}")
                    # Initialiser l'embedding function
                    model_kwargs = {'device': 'cpu'}
                    encode_kwargs = {'normalize_embeddings': False}
                    embedding_function = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                    )
                    
                    # Recréer la connexion à ChromaDB
                    import chromadb
                    vectordb = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=embedding_function
                    )
                    self.vectordb = vectordb
                    print("Database reconnection successful")
                else:
                    print("Cannot reconnect: database path unknown")
                    return "Error: Unable to connect to the knowledge base. Please try restarting the application."
            except Exception as reconnect_error:
                print(f"Database reconnection failed: {reconnect_error}")
                return "Error: Unable to connect to the knowledge base. The database may be corrupted or inaccessible. Please try reindexing the document."

        # Ensure query is not too specific about file names
        if ".pdf" in query.lower():
            query = query.replace(".pdf", "")
            print(f"Modified query (removed .pdf): {query}")
            
        # Liste de requêtes alternatives pour les définitions et concepts
        alternative_queries = []
        
        # Détection de requêtes de définition
        search_for_definition = False
        keywords = ["définition", "definition", "concept", "meaning", "signification", "qu'est-ce que", "what is", "explain"]
        for keyword in keywords:
            if keyword in query.lower():
                search_for_definition = True
                break
        
        # Générer des variantes de requêtes pour améliorer la recherche sémantique
        original_query = query.lower()
        
        # Nettoyage et simplification de la requête
        clean_query = original_query
        for term in ["définition de", "definition of", "concept de", "concept of", "signification de", "meaning of", "qu'est-ce que", "what is"]:
            if term in clean_query:
                clean_query = clean_query.replace(term, "").strip()
        
        # Ajouter la requête nettoyée
        if clean_query != original_query:
            alternative_queries.append(clean_query)
        
        # Si on cherche une définition, ajouter des variantes spécifiques
        if search_for_definition:
            term = clean_query
            # Cas spécifique pour "tokenised deposits"
            if "tokenised" in term or "tokenized" in term:
                if "deposits" in term:
                    alternative_queries.extend([
                        "tokenised deposits definition",
                        "tokenized deposits definition",
                        "definition of tokenised deposits",
                        "tokenised deposits concept",
                        "tokenized deposits concept",
                        "what are tokenised deposits",
                        "tokenised deposits explanation"
                    ])
            else:
                # Pour d'autres termes
                alternative_queries.extend([
                    f"{term} definition",
                    f"definition of {term}",
                    f"{term} concept",
                    f"what is {term}"
                ])

        # Focus on concepts rather than document names
        if "stablecoins vs tokenised deposits" in query.lower():
            query = "tokenised deposits definition concept"
            print(f"Modified query (simplified to concepts): {query}")

        # Combine query and notes if provided
        effective_query = query
        if additional_notes and isinstance(additional_notes, str) and additional_notes.strip():
            effective_query = f"{query}\n\nAdditional Context/Notes:\n{additional_notes.strip()}"
            print(f"Using combined query: {effective_query[:200]}...")
        else:
            print(f"Using simple query: {query[:200]}...")

        try:
            # Première tentative avec la requête originale
            docs = self.vectordb.similarity_search(
                effective_query,
                k=5,
            )
            print(f"Found {len(docs)} relevant documents with original query")
            
            # Si pas assez de résultats pertinents, essayer les requêtes alternatives
            if len(docs) < 2 and alternative_queries:
                print(f"Trying {len(alternative_queries)} alternative queries...")
                all_docs = docs.copy() if docs else []
                
                for alt_query in alternative_queries:
                    print(f"Trying alternative query: {alt_query}")
                    alt_docs = self.vectordb.similarity_search(
                        alt_query,
                        k=3,
                    )
                    print(f"Found {len(alt_docs)} documents with '{alt_query}'")
                    
                    # Ajouter les documents uniques
                    for doc in alt_docs:
                        # Vérification simple pour éviter les doublons
                        is_duplicate = False
                        for existing_doc in all_docs:
                            if doc.page_content[:100] == existing_doc.page_content[:100]:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_docs.append(doc)
                
                # Mettre à jour docs avec tous les résultats uniques
                if all_docs:
                    docs = all_docs[:7]  # Limiter à 7 documents au maximum
                    print(f"Combined search results: {len(docs)} unique documents")
            
            # Si toujours aucun document trouvé, essayer une recherche MMR (Maximum Marginal Relevance)
            # Cette approche favorise la diversité des résultats
            if len(docs) < 2:
                print("Using MMR search for diversity...")
                try:
                    mmr_docs = self.vectordb.max_marginal_relevance_search(
                        effective_query,
                        k=5,
                        fetch_k=10,  # Récupère 10 documents puis sélectionne les 5 plus divers
                        lambda_mult=0.5  # Équilibre entre pertinence et diversité
                    )
                    # Ajouter uniquement les documents uniques
                    for doc in mmr_docs:
                        # Vérification simple pour éviter les doublons
                        is_duplicate = False
                        for existing_doc in docs:
                            if doc.page_content[:100] == existing_doc.page_content[:100]:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            docs.append(doc)
                    print(f"MMR search added {len(docs)} unique documents")
                except Exception as mmr_error:
                    print(f"MMR search failed: {mmr_error}")
            
            # Si toujours pas de résultats pertinents, recherche par mots-clés directement dans les documents
            if len(docs) < 2:
                print("Falling back to keyword search...")
                terms_to_search = []
                if "tokenised" in clean_query or "tokenized" in clean_query:
                    terms_to_search = ["tokenised deposits", "tokenized deposits", "deposits", "tokenisation"]
                elif "cbdc" in clean_query:
                    terms_to_search = ["cbdc", "central bank digital currency", "digital currency"]
                elif "stablecoin" in clean_query:
                    terms_to_search = ["stablecoin", "stable coin", "cryptocurrency"]
                
                if terms_to_search:
                    print(f"Searching for explicit terms: {terms_to_search}")
                    try:
                        # Récupérer tous les documents de la collection
                        all_collection_docs = self.vectordb.get()
                        if all_collection_docs and "documents" in all_collection_docs:
                            keyword_docs = []
                            for i, doc_text in enumerate(all_collection_docs["documents"]):
                                for term in terms_to_search:
                                    if term.lower() in doc_text.lower():
                                        doc_obj = {"page_content": doc_text, "metadata": all_collection_docs["metadatas"][i] if "metadatas" in all_collection_docs else {}}
                                        keyword_docs.append(doc_obj)
                                        break
                            
                            # Ajouter uniquement les documents uniques
                            for doc in keyword_docs:
                                # Vérification simple pour éviter les doublons
                                is_duplicate = False
                                for existing_doc in docs:
                                    if doc["page_content"][:100] == existing_doc.page_content[:100]:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    # Convertir le dictionnaire en objet Document pour compatibilité
                                    from langchain.schema import Document
                                    langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
                                    docs.append(langchain_doc)
                            
                            print(f"Keyword search added {len(keyword_docs)} unique documents")
                    except Exception as keyword_error:
                        print(f"Keyword search failed: {keyword_error}")
            
            # Si toujours aucun document trouvé, essayer une requête de secours
            if len(docs) == 0:
                backup_queries = ["tokenized deposits", "tokenised deposits", "digital deposits", "cbdc"]
                for backup_query in backup_queries:
                    print(f"No documents found. Trying backup query: {backup_query}")
                    docs = self.vectordb.similarity_search(
                        backup_query,
                        k=5,
                    )
                    print(f"Found {len(docs)} relevant documents with backup query '{backup_query}'")
                    if len(docs) > 0:
                        break
            
            # Log document details
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"Content preview: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata'):
                    print(f"Metadata: {doc.metadata}")

            if len(docs) > 0:
                # Trier les documents en plaçant en priorité ceux qui mentionnent explicitement les termes recherchés
                key_terms = clean_query.split()
                
                def doc_relevance_score(doc):
                    score = 0
                    content = doc.page_content.lower()
                    
                    # Vérifier la présence des termes clés dans le document
                    for term in key_terms:
                        if term in content:
                            score += 10
                            # Bonus supplémentaire si le terme apparaît plusieurs fois
                            occurrences = content.count(term)
                            if occurrences > 1:
                                score += min(occurrences * 2, 10)  # Plafonné à 10 points supplémentaires
                    
                    # AMÉLIORÉ: Meilleure détection des définitions
                    definition_patterns = [
                        # Patterns anglais
                        r"tokenised deposits? (is|are|refers to|can be defined as|means)",
                        r"tokenized deposits? (is|are|refers to|can be defined as|means)",
                        r"(is|are|refers to|can be defined as|means) tokenised deposits",
                        r"(is|are|refers to|can be defined as|means) tokenized deposits",
                        r"definition of tokenised deposits",
                        r"definition of tokenized deposits",
                        r"tokenised deposits (represent|constitute)",
                        r"tokenized deposits (represent|constitute)",
                        # Patterns français
                        r"les? dépôts? tokenisés? (est|sont|fait référence à|peut être défini comme|signifie)",
                        r"(est|sont|fait référence à|peut être défini comme|signifie) les? dépôts? tokenisés?",
                        r"définition des? dépôts? tokenisés?",
                        r"les? dépôts? tokenisés? (représente|constitue)"
                    ]
                    
                    import re
                    for pattern in definition_patterns:
                        if re.search(pattern, content):
                            score += 50  # Bonus important pour les définitions explicites
                            print(f"Found definition pattern: '{pattern}' in document")
                            break
                    
                    # Bonus pour les documents qui contiennent des phrases complètes liées à la définition
                    definition_phrases = [
                        "tokenised deposits are", "tokenized deposits are", 
                        "definition of tokenised", "tokenised deposits refer to",
                        "les dépôts tokenisés sont", "définition des dépôts tokenisés"
                    ]
                    for phrase in definition_phrases:
                        if phrase in content:
                            score += 20
                    
                    # Vérifier la présence de termes explicatifs
                    explanation_terms = ["meaning", "concept", "explained", "refers to", "signification", "concept", "expliqué"]
                    for term in explanation_terms:
                        if term in content:
                            score += 5
                    
                    # Vérifier si le document contient du contenu structuré comme un glossaire
                    if "glossary" in content or "glossaire" in content or re.search(r"[A-Z][a-z]+: ", content):
                        score += 15
                    
                    # Donner un score plus élevé aux paragraphes plus courts qui sont plus susceptibles de contenir des définitions concises
                    if len(content) < 500 and ("deposit" in content or "dépôt" in content):
                        score += 10
                    
                    # Donner un score plus élevé aux paragraphes qui contiennent des citations
                    if re.search(r"\([0-9]{4}\)", content) or re.search(r"\[[0-9]+\]", content):
                        score += 5
                    
                    return score
                
                # Trier les documents par pertinence
                sorted_docs = sorted(docs, key=doc_relevance_score, reverse=True)
                
                # Ajouter des séparateurs plus visibles et du contexte dans les résultats
                results_with_context = []
                for i, doc in enumerate(sorted_docs):
                    # Extraire des informations de métadonnées si disponibles
                    metadata_info = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        if 'filename' in doc.metadata:
                            metadata_info += f"Source: {doc.metadata['filename']}"
                        if 'page' in doc.metadata:
                            metadata_info += f", Page: {doc.metadata['page']}"
                    
                    # Ajouter le document avec des séparateurs plus clairs et du contexte
                    results_with_context.append(
                        f"===== Document {str(i+1)} {metadata_info} =====\n{doc.page_content}"
                    )
                
                result = "\nRetrieved documents:\n" + "\n\n".join(results_with_context)
                
                # Ajouter une analyse des résultats trouvés
                if "tokenised deposit" in clean_query.lower() or "tokenized deposit" in clean_query.lower():
                    result += "\n\nAnalyse: Les résultats ci-dessus peuvent contenir des définitions ou explications sur les 'tokenised deposits'. Recherchez des phrases comme 'tokenised deposits are...' ou 'refers to...' qui pourraient indiquer des définitions formelles."
                
                return result
            else:
                return "No relevant documents found in the knowledge base."

        except Exception as e:
            error_msg = f"Error in RetrieverTool: {str(e)}"
            print(error_msg)
            
            # Diagnostic plus détaillé
            try:
                import sqlite3
                import glob
                
                print("Diagnostic détaillé de l'erreur SQLite:")
                
                # Vérifier si c'est une erreur SQLite
                if "sqlite" in str(e).lower():
                    print("Erreur SQLite détectée")
                    
                    # Récupérer le chemin de la base
                    persist_directory = getattr(self.vectordb, '_persist_directory', None)
                    if persist_directory:
                        print(f"Base de données située à: {persist_directory}")
                        
                        # Vérifier si le répertoire existe
                        if os.path.exists(persist_directory):
                            print(f"Le répertoire existe: {persist_directory}")
                            
                            # Lister les fichiers
                            all_files = os.listdir(persist_directory)
                            print(f"Fichiers dans le répertoire: {all_files}")
                            
                            # Chercher les fichiers SQLite
                            sqlite_files = [f for f in all_files if f.endswith('.sqlite3')]
                            print(f"Fichiers SQLite: {sqlite_files}")
                            
                            # Tenter d'ouvrir la base SQLite directement
                            for db_file in sqlite_files:
                                db_path = os.path.join(persist_directory, db_file)
                                try:
                                    print(f"Test de connexion directe à {db_path}")
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    cursor.execute("PRAGMA integrity_check")
                                    integrity = cursor.fetchone()[0]
                                    print(f"Intégrité de la base: {integrity}")
                                    conn.close()
                                except Exception as sql_e:
                                    print(f"Erreur lors du test direct: {sql_e}")
                                    
                                    # Vérifier les permissions
                                    try:
                                        import stat
                                        st_mode = os.stat(db_path).st_mode
                                        print(f"Permissions du fichier: {stat.filemode(st_mode)} ({st_mode})")
                                    except Exception as perm_e:
                                        print(f"Erreur lors de la vérification des permissions: {perm_e}")
                    else:
                        print("Impossible de déterminer le chemin de la base de données")
            except Exception as diag_e:
                print(f"Erreur lors du diagnostic: {diag_e}")
            
            return f"Error retrieving documents: {str(e)}"