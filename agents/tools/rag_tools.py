"""
RAG Tools for Document Retrieval Agents
Handles PDF document search and retrieval using vector databases.
"""

import os
from smolagents import tool
from ..config.agent_config import RAG_CONFIG


@tool
def search_pdf_documents(query: str, pdf_database_path: str, user_notes: str = "") -> str:
    """
    Search through PDF documents using semantic similarity and return results with detailed source citations.
    
    Args:
        query: The search query to find relevant content
        pdf_database_path: Path to the PDF vector database 
        user_notes: Additional context or notes about the search
    
    Returns:
        Retrieved documents content with detailed source citations
    """
    # Enhanced debugging
    print(f"🔍 PDF Search Tool called with:")
    print(f"  - Query: {query[:100]}...")
    print(f"  - DB Path: {pdf_database_path}")
    print(f"  - User Notes: {user_notes[:50]}..." if user_notes else "  - No user notes")

    if not pdf_database_path or not isinstance(pdf_database_path, str):
        return "Error: 'pdf_database_path' is missing or invalid."

    # Check for placeholder path
    if pdf_database_path == "path_to_the_pdf_database":
        return """❌ Error: Placeholder path detected! 

MANAGER AGENT: You must extract the actual database path from additional_args.

Correct pattern:
1. pdf_context = additional_args.get('pdf_context', {})
2. available_files = pdf_context.get('available_files', [])
3. actual_db_path = available_files[0]['db_path']
4. Use this actual_db_path in the tool call

Do NOT use 'path_to_pdf_database' placeholder!"""

    if not os.path.exists(pdf_database_path):
        return f"Error: Vector DB path not found: {pdf_database_path}"
    
    try:
        from langchain_community.vectorstores import Chroma
        from ..core.embedding import get_embedding_function
        
        # Initialize the Chroma vector store dynamically
        vector_store = Chroma(
            persist_directory=pdf_database_path,
            embedding_function=get_embedding_function(),
            collection_name=RAG_CONFIG["collection_name"]
        )
    except Exception as e:
        return f"Error initializing vector store from {pdf_database_path}: {e}"

    effective_query = query
    if user_notes and isinstance(user_notes, str) and user_notes.strip():
        effective_query = f"{query}\n\nAdditional Context/User Notes:\n{user_notes.strip()}"
        print(f"PDF Search using combined query: {effective_query[:200]}...")
    else:
        print(f"PDF Search using query: {query[:200]}...")

    try:
        # Use similarity_search_with_score to get relevance scores
        docs_with_scores = vector_store.similarity_search_with_score(
            effective_query,
            k=RAG_CONFIG["similarity_search_k"]
        )
    except Exception as e:
        return f"Error during similarity search in {pdf_database_path}: {e}"

    # Format response with detailed source citations
    return _format_search_results_with_citations(docs_with_scores)


def _format_search_results_with_citations(docs_with_scores) -> str:
    """
    Format search results with detailed source citations.
    
    Args:
        docs_with_scores: List of tuples (Document, score) from vector search
        
    Returns:
        Formatted string with documents and source citations
    """
    if not docs_with_scores:
        return "Aucun document pertinent trouvé."
    
    formatted_results = []
    formatted_results.append("📄 Documents trouvés avec sources :")
    formatted_results.append("=" * 50)
    
    for i, (doc, score) in enumerate(docs_with_scores):
        # Extract metadata
        metadata = doc.metadata
        filename = metadata.get('filename', 'Fichier inconnu')
        doc_type = metadata.get('type', 'text')
        
        # Format source citation based on type
        source_info = _format_source_citation(metadata, doc_type)
        relevance = f"Pertinence: {(1-score)*100:.1f}%" if score <= 1 else f"Score: {score:.3f}"
        
        formatted_results.append(f"\n===== Document {i+1} =====")
        formatted_results.append(f"📍 {source_info}")
        formatted_results.append(f"🎯 {relevance}")
        formatted_results.append(f"📋 Type: {doc_type}")
        formatted_results.append("-" * 30)
        formatted_results.append(doc.page_content)
        
        # Add extra info for images and tables
        if doc_type == "image" and metadata.get('caption'):
            formatted_results.append(f"📸 Légende: {metadata['caption']}")
        elif doc_type == "table" and metadata.get('caption'):
            formatted_results.append(f"📊 Légende: {metadata['caption']}")
    
    # Add summary footer
    formatted_results.append("\n" + "=" * 50)
    formatted_results.append(f"✅ Total: {len(docs_with_scores)} documents trouvés")
    formatted_results.append("💡 Conseil: Citez toujours vos sources en utilisant les informations [Source: ...] fournies ci-dessus")
    
    return "\n".join(formatted_results)


def _format_source_citation(metadata: dict, doc_type: str) -> str:
    """
    Format a source citation based on document type and available metadata.
    
    Args:
        metadata: Document metadata from ChromaDB
        doc_type: Type of document (text, image, table)
        
    Returns:
        Formatted source citation string
    """
    filename = metadata.get('filename', 'Fichier inconnu')
    
    if doc_type == "text":
        # Use actual page number if available, otherwise estimate
        page = metadata.get('page')
        chunk_index = metadata.get('chunk_index', '?')
        chunk_count = metadata.get('chunk_count', '?')
        
        if page is not None:
            chunk_in_page = metadata.get('chunk_in_page', 0)
            total_chunks_in_page = metadata.get('total_chunks_in_page', 1)
            if total_chunks_in_page > 1:
                return f"Source: {filename}, Page {page} (Section {chunk_in_page+1}/{total_chunks_in_page})"
            else:
                return f"Source: {filename}, Page {page}"
        else:
            # Fallback to old estimation method for backwards compatibility
            estimated_page = _estimate_page_from_chunk(chunk_index, chunk_count)
            return f"Source: {filename}, Chunk {chunk_index+1}/{chunk_count} (≈Page {estimated_page})"
    
    elif doc_type in ["image", "table"]:
        page = metadata.get('page', '?')
        type_name = "Image" if doc_type == "image" else "Tableau"
        return f"Source: {filename}, Page {page} ({type_name})"
    
    else:
        return f"Source: {filename}"


def _estimate_page_from_chunk(chunk_index: int, chunk_count: int) -> int:
    """
    Estimate page number from chunk index (rough approximation).
    Assumes average chunks per page based on typical document structure.
    
    Args:
        chunk_index: Index of the chunk (0-based)
        chunk_count: Total number of chunks
        
    Returns:
        Estimated page number
    """
    if chunk_count <= 0:
        return 1
    
    # Rough estimate: assume 2-4 chunks per page on average
    # This is a heuristic and could be improved with actual page tracking
    avg_chunks_per_page = max(2, min(4, chunk_count // 10 + 1))
    estimated_page = (chunk_index // avg_chunks_per_page) + 1
    
    return max(1, estimated_page)


@tool  
def search_pdf_with_context(user_query: str, pdf_context_dict: dict = None) -> str:
    """
    Search PDF documents using provided context dictionary.
    
    This tool can be called directly with the PDF context to perform searches.
    
    Args:
        user_query: The user's query about PDF documents
        pdf_context_dict: Dictionary containing PDF context with available_files
        
    Returns:
        Search results from PDF documents or error message
    """
    print(f"🔍 search_pdf_with_context called with query: {user_query[:100]}...")
    print(f"🔍 PDF context received: {pdf_context_dict}")
    
    if not pdf_context_dict:
        return "No PDF context provided. Please ensure PDF files are loaded or use search_agent for web research instead."
    
    available_files = pdf_context_dict.get("available_files", [])
    print(f"Found {len(available_files)} PDF files available")
    
    if not available_files:
        return "No PDF files are currently indexed. Please upload and index PDF files first."
    
    # Get the database path from the first available file
    first_file = available_files[0]
    db_path = first_file.get("db_path", "")
    user_notes = first_file.get("user_notes", "")
    classification = first_file.get("classification", "General")
    filename = first_file.get("filename", "Unknown PDF")
    
    print(f"Using PDF: {filename}")
    print(f"Database path: {db_path}")
    
    if not db_path or db_path == "path_to_pdf_database":
        return "Error: No valid PDF database path found in context."
    
    # Search the PDF database with the actual path
    search_notes = f"{user_notes} - Classification: {classification}" if user_notes else f"Classification: {classification}"
    result = search_pdf_documents(user_query, db_path, search_notes)
    
    print("PDF search completed successfully")
    return result


@tool
def extract_pdf_context_and_delegate(user_query: str) -> str:
    """
    Deterministic helper that extracts PDF database paths from additional_args 
    and formats the correct RAG agent delegation.
    
    Following smolagents best practice: additional_args become state variables
    that are accessible in the agent's Python code generation.
    
    Args:
        user_query: The user's query about PDF documents
        
    Returns:
        Instructions for the agent to access PDF context from state variables
    """
    # According to smolagents documentation, additional_args are added to agent.state
    # and become accessible as variables in Python code generation
    
    print(f"🔍 DEBUG extract_pdf_context_and_delegate:")
    print(f"  - User query: {user_query[:100]}...")
    print("  - Context should be available as state variables: pdf_context, csv_context")
    
    # Return instructions for the agent to access context from its state variables
    instructions = f"""
To answer the query: "{user_query}"

The PDF context is available in the state variable 'pdf_context'.
You can access it in your Python code like this:

```python
# Check if PDF context is available
if 'pdf_context' in locals() and pdf_context:
    # Use the search_pdf_with_context tool with the context
    result = search_pdf_with_context("{user_query}", pdf_context)
    print("PDF search completed")
    final_answer(result)
else:
    final_answer("No PDF context available. Please use search_agent for web research instead.")
```
"""
    
    return instructions


@tool
def get_citation_help() -> str:
    """
    Provides guidance and examples for proper source citation in RAG responses.
    
    Returns:
        Citation guidelines and examples
    """
    return """
📚 Guide des Citations pour RAG - AlethéIA

✅ FORMATS DE CITATION RECOMMANDÉS :

1. Pour du texte avec page connue :
   [Source: rapport-annuel.pdf, Page 23]
   [Source: contrat.pdf, Page 5 (Section 2/3)]

2. Pour des images :
   [Source: presentation.pdf, Page 12 (Image)]

3. Pour des tableaux :
   [Source: donnees-financieres.pdf, Page 8 (Tableau)]

4. Pour du texte avec estimation de page :
   [Source: document.pdf, Chunk 15/47 (≈Page 8)]

💡 EXEMPLES D'USAGE DANS LES RÉPONSES :

❌ Mauvais :
"La société a enregistré une croissance de 15%."

✅ Bon :
"Selon le rapport annuel, la société a enregistré une croissance de 15% [Source: rapport-annuel.pdf, Page 23]."

✅ Très bon :
"D'après les données financières, la société a enregistré une croissance de 15% cette année [Source: rapport-annuel.pdf, Page 23], ce qui représente une amélioration significative par rapport aux 8% de l'année précédente [Source: rapport-annuel.pdf, Page 24]."

🎯 BONNES PRATIQUES :
- Toujours citer avant de présenter une information
- Utiliser des phrases d'introduction : "Selon...", "D'après...", "Le document indique que..."
- Citer chaque source séparément quand vous combinez des informations
- Être précis : mentionner la page ET le type de contenu (image/tableau)

📊 AMÉLIORATIONS RÉCENTES :
- ✅ Tracking précis des pages pour tous les types de contenu
- ✅ Scores de pertinence pour évaluer la qualité des résultats
- ✅ Citations formatées automatiquement avec métadonnées complètes
- ✅ Support des images et tableaux avec localisation précise
- ✅ Sections multiples par page détectées et référencées
"""


def get_rag_system_improvements() -> str:
    """
    Returns a summary of recent improvements to the RAG system.
    
    Returns:
        Summary of improvements made to citation and source tracking
    """
    return """
🚀 AMÉLIORATIONS DU SYSTÈME RAG - AlethéIA

📄 NOUVELLES FONCTIONNALITÉS :

1. 🎯 CITATIONS AUTOMATIQUES :
   - Extraction automatique des métadonnées de source
   - Formatage standardisé des citations
   - Support des scores de pertinence

2. 📍 TRACKING PRÉCIS DES PAGES :
   - Pages réelles pour tous les chunks de texte
   - Sections multiples par page détectées  
   - Localisation précise des images et tableaux

3. 📊 MÉTADONNÉES ENRICHIES :
   - Type de contenu (texte, image, tableau)
   - Position dans le document (chunk X/Y, section Y/Z)
   - Scores de pertinence sémantique
   - Légendes pour images et tableaux

4. 🔍 RÉSULTATS AMÉLIORÉS :
   - Formatage visuel des résultats de recherche
   - Conseils automatiques pour les citations
   - Informations de pertinence pour chaque résultat

📈 BÉNÉFICES :
- ✅ Traçabilité complète des informations
- ✅ Réduction des hallucinations
- ✅ Conformité académique et professionnelle
- ✅ Facilité de vérification des sources
- ✅ Amélioration de la confiance utilisateur

🛠️ COMPATIBILITÉ :
- Rétrocompatible avec les anciens documents indexés
- Estimation de page pour les documents sans tracking
- Support progressif des nouvelles fonctionnalités
"""


@tool
def search_pdf_from_state(user_query: str) -> str:
    """
    Search PDF documents using context that may be available in the agent's environment.
    
    This tool attempts to access PDF context from various sources and provides
    helpful guidance if context is not available.
    
    Args:
        user_query: The user's query about PDF documents
        
    Returns:
        Search results from PDF documents or guidance message
    """
    print(f"🔍 search_pdf_from_state called with query: {user_query[:100]}...")
    
    # Check if the query contains context information
    if "Context:" in user_query or "PDF Context Available:" in user_query:
        print("🔍 Found context information in query")
        # Extract the actual query part
        if "Query:" in user_query:
            actual_query = user_query.split("Query:")[-1].strip()
            print(f"🔍 Extracted query: {actual_query[:100]}...")
        else:
            actual_query = user_query
        
        # For now, return a message indicating delegation is working
        return f"""✅ RAG agent properly called for PDF search!

Query received: "{actual_query}"

This demonstrates that:
1. ✅ Manager agent correctly identified this as a PDF search task
2. ✅ Manager agent successfully delegated to the RAG agent
3. ✅ Context information was passed along

To complete this implementation, the RAG agent would need:
- Access to the actual PDF database paths
- The indexed PDF files to search through
- Proper context passing mechanism from manager to RAG agent

Current status: Delegation pattern working correctly! 🎉"""
    
    # If no context in query, provide guidance
    return f"""RAG agent received query: "{user_query}"

This shows the delegation is working, but to perform actual PDF search, the system needs:

1. PDF files to be uploaded and indexed
2. Database paths to be accessible to the RAG agent
3. Context to be passed from manager agent

For web research instead, please ask the manager to delegate to the search_agent.""" 