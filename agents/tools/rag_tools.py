"""
RAG Tools for Document Retrieval Agents - Streamlit Optimized
Handles PDF document search and retrieval using vector databases with interactive Streamlit display.
Following smolagents best practices for clear descriptions, error handling, and user guidance.
"""

import os
import json
import re
from smolagents import tool
from ..config.agent_config import RAG_CONFIG


@tool
def search_pdf_for_streamlit(query: str, pdf_database_path: str, user_notes: str = "") -> str:
    """
    Search PDF documents and return results optimized for Streamlit display.
    
    This tool returns structured JSON data that can be easily parsed and displayed
    in a Streamlit interface with enhanced formatting, expandable sections, and
    interactive elements.
    
    Args:
        query: The search query to find relevant content in PDF documents
        pdf_database_path: Path to the vector database containing indexed PDF documents
        user_notes: Additional context or notes about the search
    
    Returns:
        JSON string containing structured search results with metadata for Streamlit display
    """
    
    # Input validation
    if not query or not isinstance(query, str):
        return json.dumps({
            "success": False,
            "error": "Query must be a non-empty string",
            "error_type": "validation_error"
        })

    if not pdf_database_path or not isinstance(pdf_database_path, str):
        return json.dumps({
            "success": False,
            "error": "PDF database path is missing or invalid",
            "error_type": "validation_error"
        })

    # Check for placeholder path
    if pdf_database_path == "path_to_the_pdf_database":
        return json.dumps({
            "success": False,
            "error": "Placeholder path detected. Manager agent must extract actual database path.",
            "error_type": "placeholder_error",
            "guidance": "Use pdf_context from additional_args to get the real database path"
        })

    # Verify database path exists
    if not os.path.exists(pdf_database_path):
        return json.dumps({
            "success": False,
            "error": f"Vector database path not found: {pdf_database_path}",
            "error_type": "path_error"
        })
    
    try:
        from langchain_community.vectorstores import Chroma
        from ..core.embedding import get_embedding_function
        
        # Initialize the Chroma vector store
        vector_store = Chroma(
            persist_directory=pdf_database_path,
            embedding_function=get_embedding_function(),
            collection_name=RAG_CONFIG["collection_name"]
        )
        
        # Perform similarity search
        search_k = RAG_CONFIG["similarity_search_k"]
        results = vector_store.similarity_search_with_score(query, k=search_k)
        
        if not results:
            return json.dumps({
                "success": True,
                "results": [],
                "query": query,
                "total_results": 0,
                "database_path": pdf_database_path,
                "user_notes": user_notes,
                "message": "Aucun document pertinent trouvé pour cette requête"
            })

        # Structure results for Streamlit
        structured_results = []
        total_content_length = 0
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata or {}
            content = doc.page_content.strip()
            content_length = len(content)
            total_content_length += content_length
            
            # Calculate relevance percentage
            relevance_percent = (1 - score) * 100 if score <= 1 else 0
            
            result_item = {
                "rank": i,
                "content": content,
                "metadata": {
                    "source": metadata.get('source', 'Document inconnu'),
                    "page": metadata.get('page', 'Page inconnue'),
                    "relevance_score": round(relevance_percent, 1),
                    "similarity_score": round(1-score, 3) if score <= 1 else round(score, 3),
                    "content_length": content_length
                },
                "preview": content[:200] + "..." if len(content) > 200 else content
            }
            structured_results.append(result_item)
        
        # Calculate average relevance
        avg_relevance = sum(r["metadata"]["relevance_score"] for r in structured_results) / len(structured_results)
        
        return json.dumps({
            "success": True,
            "results": structured_results,
            "query": query,
            "total_results": len(results),
            "database_path": pdf_database_path,
            "user_notes": user_notes,
            "summary": {
                "total_content_length": total_content_length,
                "average_relevance": round(avg_relevance, 1),
                "top_relevance": structured_results[0]["metadata"]["relevance_score"] if structured_results else 0
            }
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        return json.dumps({
            "success": False,
            "error": f"Missing dependencies: {str(e)}",
            "error_type": "import_error"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {str(e)}",
            "error_type": "search_error"
        })


@tool
def search_pdf_with_context_for_streamlit(user_query: str, pdf_context_dict: dict = None) -> str:
    """
    Search PDF documents using context and return Streamlit-optimized results.
    
    This tool combines context-aware search with structured output for enhanced
    Streamlit display capabilities.
    
    Args:
        user_query: The user's query about PDF documents
        pdf_context_dict: Dictionary containing PDF context with available_files
        
    Returns:
        JSON string with structured results optimized for Streamlit interface
    """
    
    # Input validation
    if not user_query or not isinstance(user_query, str):
        return json.dumps({
            "success": False,
            "error": "User query must be a non-empty string",
            "error_type": "validation_error"
        })
    
    if not pdf_context_dict:
        return json.dumps({
            "success": False,
            "error": "No PDF context provided",
            "error_type": "context_error",
            "guidance": "PDF context required for search operation"
        })
    
    if not isinstance(pdf_context_dict, dict):
        return json.dumps({
            "success": False,
            "error": f"PDF context must be a dictionary, received: {type(pdf_context_dict)}",
            "error_type": "context_error"
        })
    
    available_files = pdf_context_dict.get("available_files", [])
    
    if not available_files:
        return json.dumps({
            "success": False,
            "error": "No PDF files available in context",
            "error_type": "no_files_error",
            "context_structure": pdf_context_dict
        })
    
    # Get database path from first available file
    first_file = available_files[0]
    db_path = first_file.get("db_path", "")
    user_notes = first_file.get("user_notes", "")
    classification = first_file.get("classification", "General")
    filename = first_file.get("filename", "Unknown PDF")
    
    if not db_path or db_path == "path_to_pdf_database":
        return json.dumps({
            "success": False,
            "error": "No valid PDF database path found in context",
            "error_type": "path_error",
            "file_context": first_file
        })
    
    # Enhance search notes with context
    search_notes = f"{user_notes} - Classification: {classification}" if user_notes else f"Classification: {classification}"
    
    try:
        # Use the Streamlit-optimized search function
        result_json = search_pdf_for_streamlit(user_query, db_path, search_notes)
        result_data = json.loads(result_json)
        
        # Enhance with context information
        if result_data["success"]:
            result_data["context"] = {
                "filename": filename,
                "classification": classification,
                "user_notes": user_notes,
                "total_available_files": len(available_files)
            }
        
        return json.dumps(result_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Context search failed: {str(e)}",
            "error_type": "search_error",
            "context_info": {
                "filename": filename,
                "db_path": db_path,
                "classification": classification
            }
        })


@tool
def smart_pdf_search_for_streamlit(user_query: str) -> str:
    """
    Intelligent PDF search with automatic context detection, optimized for Streamlit.
    
    This tool provides the best user experience by automatically detecting available
    PDF context and returning structured results that can be beautifully displayed
    in a Streamlit interface.
    
    Args:
        user_query: The search query for PDF documents
        
    Returns:
        JSON string with structured search results or guidance for Streamlit display
    """
    
    if not user_query or not isinstance(user_query, str):
        return json.dumps({
            "success": False,
            "error": "Please provide a valid search query",
            "error_type": "validation_error"
        })
    
    # Try to get context from session state
    try:
        import json as json_module
        import os
        
        session_file = "data/session_persistence/persistent_session_state.json"
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                processed_files = json_module.load(f)
            
            # Build PDF context
            available_pdfs_context = []
            for fid, details in processed_files.items():
                if details.get('type') == 'pdf' and details.get('indexed') and details.get('db_path'):
                    available_pdfs_context.append({
                        'file_id': fid,
                        'filename': details.get('filename', 'Unknown PDF'),
                        'classification': details.get('classification'),
                        'db_path': details.get('db_path'),
                        'user_notes': details.get('user_notes', ''),
                        'summary': details.get('summary', '')
                    })
            
            if available_pdfs_context:
                pdf_context = {
                    "available_files": available_pdfs_context,
                    "count": len(available_pdfs_context),
                    "classifications": list(set(
                        pdf.get('classification', 'General') 
                        for pdf in available_pdfs_context
                    ))
                }
                
                return search_pdf_with_context_for_streamlit(user_query, pdf_context)
        
    except Exception as e:
        pass  # Continue to fallback
    
    # No context found - return guidance
    return json.dumps({
        "success": False,
        "error": "Aucun document PDF disponible actuellement",
        "error_type": "no_context",
        "guidance": {
            "message": "Pour utiliser la recherche dans les documents PDF, vous devez :",
            "steps": [
                "1. Télécharger des fichiers PDF via l'interface",
                "2. Classifier les documents dans la barre latérale",
                "3. Indexer les PDFs pour la recherche",
                "4. Puis poser vos questions sur le contenu"
            ],
            "alternative": f"Je peux rechercher des informations sur '{user_query}' en ligne si vous le souhaitez."
        },
        "query": user_query
    }, ensure_ascii=False, indent=2)


@tool
def search_pdf_interactive(user_query: str) -> str:
    """
    Recherche dans les documents PDF avec affichage style NotebookLM.
    
    Génère une réponse avec citations numérotées [1], [2] intégrées directement 
    dans le texte, suivies des sources correspondantes.
    
    Args:
        user_query: La requête de recherche pour les documents PDF
        
    Returns:
        Réponse avec citations intégrées + JSON des sources pour l'affichage NotebookLM
    """
    
    if not user_query or not isinstance(user_query, str):
        return "❌ Veuillez fournir une requête de recherche valide."
    
    print(f"🔍 Recherche PDF style NotebookLM pour: '{user_query}'")
    
    # Effectue la recherche
    results_json = smart_pdf_search_for_streamlit(user_query)
    
    try:
        import json
        # Analyse les résultats
        results = json.loads(results_json)
        
        if results.get("success"):
            print("✅ Recherche réussie, génération réponse NotebookLM...")
            
            # Génère une réponse avec citations intégrées
            response_with_citations = _generate_notebooklm_response(results, user_query)
            
            # Combine la réponse et les sources pour l'affichage
            combined_response = f"{response_with_citations}\n\n```json\n{results_json}\n```"
            
            return combined_response
        
        else:
            print("❌ Recherche échouée...")
            
            error_type = results.get("error_type", "unknown")
            if error_type == "no_context":
                return """📚 **Aucun document PDF disponible**

💡 **Pour rechercher dans vos documents :**
1. Chargez un fichier PDF via la barre latérale
2. Attendez l'indexation automatique
3. Posez votre question !

🌐 Je peux aussi faire une recherche web si vous préférez."""
            
            else:
                return f"""❌ **Problème de recherche :** {results.get('error', 'Erreur inconnue')}

🛠️ **Essayez de :**
- Reformuler votre question
- Vérifier que le document contient l'information
- Recharger le fichier PDF"""
            
    except Exception as e:
        print(f"❌ Erreur dans search_pdf_interactive: {e}")
        return f"❌ Erreur lors de la recherche : {str(e)}"


def _generate_notebooklm_response(results: dict, user_query: str) -> str:
    """
    Génère une réponse style NotebookLM avec citations numérotées intégrées.
    
    Args:
        results: Résultats de recherche parsés
        user_query: Requête originale de l'utilisateur
        
    Returns:
        Réponse narrative avec citations [1], [2], etc.
    """
    search_results = results.get("results", [])
    context = results.get("context", {})
    
    if not search_results:
        return "Aucune information pertinente trouvée dans les documents."
    
    # Informations du document
    doc_name = context.get("filename", "").replace('.pdf', '')
    total_results = len(search_results)
    
    # Construction de la réponse avec citations
    response_parts = []
    
    # Introduction
    if doc_name:
        response_parts.append(f"Selon le document **{doc_name}**, voici les informations pertinentes concernant votre recherche :")
        response_parts.append("")
    
    # Synthèse des informations avec citations
    for i, result in enumerate(search_results[:3], 1):  # Limite à 3 sources principales
        content = result.get("content", "")
        metadata = result.get("metadata", {})
        page = metadata.get("page", "?")
        
        # Extrait les points clés du contenu
        key_points = _extract_key_information(content, user_query)
        
        if key_points:
            # Intègre la citation numérotée
            response_parts.append(f"{key_points} [{i}]")
            response_parts.append("")
    
    # Conclusion si plusieurs sources
    if total_results > 1:
        response_parts.append(f"Ces informations proviennent de {total_results} passages pertinents du document.")
        if total_results > 3:
            response_parts.append(f"Les sources détaillées ci-dessous incluent {total_results-3} passages supplémentaires.")
    
    return "\n".join(response_parts)


def _extract_key_information(content: str, query: str) -> str:
    """
    Extrait l'information clé du contenu en relation avec la requête.
    
    Args:
        content: Contenu du passage
        query: Requête originale
        
    Returns:
        Information reformulée de manière naturelle
    """
    if not content:
        return ""
    
    # Nettoie le contenu
    cleaned_content = content.strip()
    
    # Pour une réponse naturelle, on prend la phrase la plus pertinente
    sentences = []
    for sep in ['. ', '! ', '? ']:
        if sep in cleaned_content:
            sentences = cleaned_content.split(sep)
            break
    
    if not sentences:
        # Si pas de phrases distinctes, prend le début
        return cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
    
    # Sélectionne les phrases les plus pertinentes (simple heuristique)
    query_words = set(query.lower().split())
    best_sentences = []
    
    for sentence in sentences[:3]:  # Limite aux 3 premières phrases
        sentence = sentence.strip()
        if len(sentence) > 20:  # Ignore les phrases trop courtes
            # Score simple basé sur les mots de la requête
            sentence_words = set(sentence.lower().split())
            relevance = len(query_words.intersection(sentence_words))
            
            if relevance > 0 or not best_sentences:  # Prend au moins une phrase
                best_sentences.append((sentence, relevance))
    
    # Trie par pertinence et prend les meilleures
    best_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Construit la réponse
    if best_sentences:
        # Prend les 2 meilleures phrases
        selected = [s[0] for s in best_sentences[:2]]
        result = '. '.join(selected)
        
        # Assure qu'on termine par un point
        if not result.endswith(('.', '!', '?')):
            result += '.'
            
        return result
    
    # Fallback
    return cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content


@tool
def get_citation_help() -> str:
    """
    Provide guidance on properly citing PDF sources and troubleshooting RAG search issues.
    
    This tool offers comprehensive guidance on source citation best practices
    and troubleshooting steps for PDF document retrieval problems. Following
    smolagents best practices for detailed user guidance and error resolution.
    
    Returns:
        Comprehensive guidance on citation practices and troubleshooting steps
        for PDF search and retrieval operations.
        
    Usage Examples:
        - get_citation_help() - Get general guidance on citations
        - Can be called when search results need proper source attribution
        - Useful for troubleshooting PDF search problems
    """
    citation_guidance = """
📚 PDF CITATION AND TROUBLESHOOTING GUIDE

✅ PROPER SOURCE CITATION:
When referencing PDF search results, always include:

1. **Document Title/Source**: Name of the PDF file
2. **Page Reference**: Specific page number where information was found
3. **Relevance Score**: How well the content matches the search query
4. **Content Excerpt**: Brief quote or summary of the relevant content

📖 CITATION FORMAT EXAMPLE:
"According to the ACPR recommendations (Source: ACPR_Guidelines_2024.pdf, Page 15, 92.3% relevance), 
the key principle for wealth management is..."

🔍 SEARCH QUALITY INDICATORS:
- **High Relevance (>80%)**: Strong semantic match, very reliable
- **Medium Relevance (60-80%)**: Good match, generally reliable
- **Low Relevance (<60%)**: Weak match, use with caution

🛠️ TROUBLESHOOTING PDF SEARCH PROBLEMS:

**Problem 1: "No PDF context available"**
- Check if PDFs were properly uploaded and indexed
- Verify the agent has access to state variables
- Try using search_pdf_interactive() tool

**Problem 2: "Vector database path not found"**
- Check if PDF indexing completed successfully
- Verify database files exist in the expected location
- Try re-uploading and re-indexing the PDF

**Problem 3: "No relevant documents found"**
- Try broader or different search keywords
- Check if the document contains the information sought
- Use synonyms or related terms
- Break complex queries into simpler parts

**Problem 4: "Low relevance scores"**
- Refine search query to be more specific
- Try multiple related queries
- Check if documents are in the expected language
- Consider that the information might not be in the PDFs

💡 SEARCH OPTIMIZATION TIPS:
- Use specific keywords rather than full sentences
- Include domain-specific terminology
- Try both French and English terms if applicable
- Use "OR" logic by running multiple searches

🆘 EMERGENCY FALLBACKS:
If PDF search fails:
1. Try searching with different keywords
2. Use web search tools for similar information
3. Ask user to clarify what specific information they need
4. Suggest checking if the right documents are uploaded

📞 SUPPORT ESCALATION:
If problems persist:
- Document the exact error messages
- Note what PDF files are supposedly available
- Check the context variables and database paths
- Contact technical support with specific details
"""
    
    print("📖 Providing PDF citation and troubleshooting guidance")
    return citation_guidance


@tool
def diagnose_pdf_context() -> str:
    """
    Diagnostic tool to check PDF context availability and troubleshoot access issues.
    
    This tool helps agents and users understand the current state of PDF context,
    database availability, and common issues that might prevent document access.
    Following smolagents best practices for comprehensive diagnostic reporting.
    
    Returns:
        Detailed diagnostic report of PDF context status, available files,
        database paths, and troubleshooting recommendations.
        
    Usage Examples:
        - diagnose_pdf_context() - Check overall PDF system status
        - Use when PDF searches are failing
        - Call before attempting document searches
    """
    
    print("🔍 Running PDF context diagnostics...")
    
    diagnostic_report = []
    diagnostic_report.append("🏥 PDF CONTEXT DIAGNOSTIC REPORT")
    diagnostic_report.append("=" * 50)
    diagnostic_report.append("")
    
    # Check for state variables
    context_available = False
    pdf_context = None
    
    try:
        # Check if pdf_context is available in local scope
        import sys
        frame = sys._getframe(1)  # Get the calling frame
        local_vars = frame.f_locals
        global_vars = frame.f_globals
        
        if 'pdf_context' in local_vars:
            pdf_context = local_vars['pdf_context']
            context_available = True
            diagnostic_report.append("✅ PDF context found in local variables")
        elif 'pdf_context' in global_vars:
            pdf_context = global_vars['pdf_context']
            context_available = True
            diagnostic_report.append("✅ PDF context found in global variables")
        else:
            diagnostic_report.append("❌ No PDF context found in local or global variables")
            
    except Exception as e:
        diagnostic_report.append(f"⚠️ Error checking context variables: {str(e)}")
    
    diagnostic_report.append("")
    
    if context_available and pdf_context:
        diagnostic_report.append("📊 CONTEXT ANALYSIS:")
        diagnostic_report.append(f"   Type: {type(pdf_context)}")
        
        if isinstance(pdf_context, dict):
            diagnostic_report.append(f"   Available files count: {pdf_context.get('count', 'Unknown')}")
            
            available_files = pdf_context.get('available_files', [])
            if available_files:
                diagnostic_report.append("   📄 Available PDF files:")
                for i, file_info in enumerate(available_files[:5], 1):  # Show max 5 files
                    filename = file_info.get('filename', 'Unknown')
                    db_path = file_info.get('db_path', 'No path')
                    classification = file_info.get('classification', 'Unclassified')
                    diagnostic_report.append(f"     {i}. {filename}")
                    diagnostic_report.append(f"        Path: {db_path}")
                    diagnostic_report.append(f"        Category: {classification}")
                    
                    # Check if database path exists
                    import os
                    if db_path and db_path != "No path":
                        if os.path.exists(db_path):
                            diagnostic_report.append(f"        ✅ Database accessible")
                        else:
                            diagnostic_report.append(f"        ❌ Database path not found")
                    else:
                        diagnostic_report.append(f"        ⚠️ No database path configured")
                    diagnostic_report.append("")
                    
                if len(available_files) > 5:
                    diagnostic_report.append(f"     ... and {len(available_files) - 5} more files")
            else:
                diagnostic_report.append("   ❌ No available files in context")
        else:
            diagnostic_report.append(f"   ⚠️ Context is not a dictionary: {pdf_context}")
    
    diagnostic_report.append("")
    diagnostic_report.append("🛠️ TROUBLESHOOTING RECOMMENDATIONS:")
    
    if not context_available:
        diagnostic_report.append("   📋 NO CONTEXT AVAILABLE:")
        diagnostic_report.append("   - Check if PDFs were uploaded and indexed")
        diagnostic_report.append("   - Verify manager agent is passing context correctly")
        diagnostic_report.append("   - Use search_pdf_interactive() tool for automatic context handling")
        diagnostic_report.append("   - Try re-uploading PDF files")
        
    elif pdf_context and isinstance(pdf_context, dict):
        available_files = pdf_context.get('available_files', [])
        if not available_files:
            diagnostic_report.append("   📋 EMPTY CONTEXT:")
            diagnostic_report.append("   - PDF context exists but contains no files")
            diagnostic_report.append("   - Check if PDF indexing completed successfully")
            diagnostic_report.append("   - Verify files were processed without errors")
        else:
            diagnostic_report.append("   📋 CONTEXT LOOKS GOOD:")
            diagnostic_report.append("   - PDF context is properly formatted")
            diagnostic_report.append("   - Files are available for searching")
            diagnostic_report.append("   - Use search_pdf_interactive() or smart_pdf_search_for_streamlit()")
    
    diagnostic_report.append("")
    diagnostic_report.append("💡 NEXT STEPS:")
    diagnostic_report.append("   1. If context is available, try search_pdf_interactive()")
    diagnostic_report.append("   2. If no context, use smart_pdf_search_for_streamlit() for automatic handling")
    diagnostic_report.append("   3. For direct access, use search_pdf_for_streamlit() with known path")
    diagnostic_report.append("   4. If all fails, suggest web search as alternative")
    
    result = "\n".join(diagnostic_report)
    print("✅ PDF context diagnostic completed")
    return result 


@tool
def display_source_passages_enhanced(search_results: str, query: str = "", response_text: str = "") -> str:
    """
    Display source passages with interactive highlighting and bidirectional links.
    
    Enhanced version that provides ChatGPT/NotebookLM-style highlighting where
    each part of the response can be linked to its source passages, and sources
    can be highlighted when hovering over response text.
    
    Args:
        search_results: JSON string containing search results with passages and metadata
        query: Original search query for context display
        response_text: Agent's response text to create bidirectional links (optional)
        
    Returns:
        Confirmation message that passages were displayed in Streamlit interface
    """
    
    print(f"🎨 Displaying enhanced source passages with interactive highlighting for query: '{query}'")
    
    try:
        import streamlit as st
        import json
        import re
        import hashlib
        import time
        from difflib import SequenceMatcher
        
        # Parse search results
        if isinstance(search_results, str):
            results_data = json.loads(search_results)
        else:
            results_data = search_results
            
        if not isinstance(results_data, dict):
            st.error("❌ Error: Search results must be a JSON dictionary")
            return "Error: Invalid search results format"
            
        if not results_data.get("success", False):
            error_msg = results_data.get("error", "Unknown error")
            st.error(f"❌ Search failed: {error_msg}")
            return f"Search failed: {error_msg}"
            
        passages = results_data.get("results", [])
        if not passages:
            st.info(f"📭 No source passages found for query: '{query}'")
            return f"No source passages found for query: '{query}'"
            
        print(f"✅ Displaying {len(passages)} passages with enhanced interactivity")
        
        # Mark that sources have been displayed in session state
        if 'st' in globals():
            st.session_state.sources_displayed_flag = True
            st.session_state.last_sources_display_time = time.time()
        
        # Generate unique session key for this display
        session_key = hashlib.md5(f"{query}_{len(passages)}".encode()).hexdigest()[:8]
        
        # Initialize session state for highlighting
        if f"highlight_state_{session_key}" not in st.session_state:
            st.session_state[f"highlight_state_{session_key}"] = {
                "selected_passage": None,
                "highlight_mode": False
            }
        
        # Custom CSS for interactive highlighting
        st.markdown("""
        <style>
        .passage-highlight {
            background: linear-gradient(120deg, #a8e6cf 0%, #dcedc8 100%);
            border-left: 4px solid #4caf50;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .passage-highlight:hover {
            background: linear-gradient(120deg, #81c784 0%, #c8e6c9 100%);
            transform: translateX(4px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .passage-highlight.selected {
            background: linear-gradient(120deg, #ffeb3b 0%, #fff59d 100%);
            border-left-color: #ff9800;
            box-shadow: 0 4px 12px rgba(255,193,7,0.3);
        }
        .source-link {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 0 2px;
            cursor: pointer;
            border: 1px solid #bbdefb;
            transition: all 0.2s ease;
        }
        .source-link:hover {
            background: #1976d2;
            color: white;
            transform: scale(1.05);
        }
        .interactive-text {
            line-height: 1.8;
            font-size: 1.1em;
        }
        .relevance-bar {
            height: 6px;
            border-radius: 3px;
            margin: 4px 0;
            transition: all 0.3s ease;
        }
        .relevance-high { background: linear-gradient(90deg, #4caf50, #8bc34a); }
        .relevance-medium { background: linear-gradient(90deg, #ff9800, #ffc107); }
        .relevance-low { background: linear-gradient(90deg, #f44336, #ff7043); }
        </style>
        """, unsafe_allow_html=True)
        
        # Header with enhanced styling
        st.markdown("### 📚 SOURCE PASSAGES WITH INTERACTIVE HIGHLIGHTING")
        st.markdown("---")
        
        # Interactive mode toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if query:
                st.markdown(f"**🔍 Search Query:** `{query}`")
        with col2:
            highlight_mode = st.toggle("🎯 Highlight Mode", 
                                     value=st.session_state[f"highlight_state_{session_key}"]["highlight_mode"],
                                     key=f"highlight_toggle_{session_key}")
            st.session_state[f"highlight_state_{session_key}"]["highlight_mode"] = highlight_mode
        with col3:
            if st.button("🔄 Reset Selection", key=f"reset_{session_key}"):
                st.session_state[f"highlight_state_{session_key}"]["selected_passage"] = None
                st.rerun()
        
        # Context and summary information
        summary = results_data.get("summary", {})
        context = results_data.get("context", {})
        
        if context or summary:
            with st.expander("📊 Search Summary", expanded=False):
                if context:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Document", context.get("filename", "Unknown").replace(".pdf", ""))
                    with col2:
                        st.metric("🏷️ Classification", context.get("classification", "N/A"))
                    with col3:
                        st.metric("📊 Found passages", len(passages))
                        
                if summary:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_relevance = summary.get("average_relevance", 0)
                        st.metric("🎯 Average relevance", f"{avg_relevance:.1f}%")
                    with col2:
                        total_content = summary.get("total_content_length", 0)
                        content_kb = total_content / 1000
                        st.metric("📏 Total content", f"{content_kb:.1f}k chars")
                    with col3:
                        top_relevance = summary.get("top_relevance", 0)
                        st.metric("⭐ Top relevance", f"{top_relevance:.1f}%")
        
        st.markdown("---")
        
        # Enhanced passage display with interactive features
        selected_passage = st.session_state[f"highlight_state_{session_key}"]["selected_passage"]
        
        for i, passage in enumerate(passages, 1):
            metadata = passage.get("metadata", {})
            content = passage.get("content", "")
            
            # Extract metadata
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown")
            relevance = metadata.get("relevance_score", 0)
            content_length = len(content)
            
            # Relevance classification and styling
            if relevance >= 80:
                relevance_color = "🟢"
                relevance_label = "Highly relevant"
                relevance_class = "relevance-high"
                color_theme = "#4caf50"
            elif relevance >= 60:
                relevance_color = "🟡"
                relevance_label = "Relevant"
                relevance_class = "relevance-medium"
                color_theme = "#ff9800"
            else:
                relevance_color = "🟠"
                relevance_label = "Less relevant"
                relevance_class = "relevance-low"
                color_theme = "#f44336"
            
            # Determine if this passage is selected
            is_selected = selected_passage == i
            highlight_class = "passage-highlight selected" if is_selected else "passage-highlight"
            
            # Interactive passage container
            passage_container = st.container()
            
            with passage_container:
                # Clickable passage header
                col1, col2 = st.columns([4, 1])
                with col1:
                    passage_header = f"{relevance_color} **Passage #{i}** - {relevance:.1f}% ({relevance_label})"
                    if st.button(passage_header, key=f"select_passage_{i}_{session_key}", 
                               use_container_width=True):
                        st.session_state[f"highlight_state_{session_key}"]["selected_passage"] = i if not is_selected else None
                        st.rerun()
                
                with col2:
                    # Quick copy button
                    if st.button("📋", key=f"quick_copy_{i}_{session_key}", 
                               help="Copy passage content"):
                        st.success("✅ Ready to copy!")
                
                # Relevance bar
                st.markdown(f'<div class="relevance-bar {relevance_class}" style="width: {relevance}%;"></div>', 
                          unsafe_allow_html=True)
                
                # Expandable detailed view
                with st.expander(
                    f"📖 Details & Content",
                    expanded=is_selected or (i <= 2 and not any(j > i for j in range(1, len(passages)+1) 
                                                               if st.session_state[f"highlight_state_{session_key}"]["selected_passage"] == j))
                ):
                    # Enhanced metadata display
                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                    with meta_col1:
                        st.markdown(f"**📁 Source:**<br/>{source}", unsafe_allow_html=True)
                    with meta_col2:
                        st.markdown(f"**📖 Page:**<br/>{page}", unsafe_allow_html=True)
                    with meta_col3:
                        st.markdown(f"**📏 Length:**<br/>{content_length} chars", unsafe_allow_html=True)
                    with meta_col4:
                        st.markdown(f"**🎯 Relevance:**<br/>{relevance:.1f}%", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Enhanced content tabs with additional features
                    tab1, tab2, tab3, tab4 = st.tabs(["📖 Full Content", "🔗 Smart Citation", "🎯 Key Excerpts", "📊 Analysis"])
                    
                    with tab1:
                        st.markdown("**Original content from PDF:**")
                        
                        # Highlighted content with keyword emphasis
                        highlighted_content = _highlight_keywords_in_content(content, query)
                        st.markdown(f'<div class="interactive-text">{highlighted_content}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Enhanced copy functionality
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📋 Copy Full Content", key=f"copy_full_{i}_{session_key}"):
                                st.success("✅ Content ready to copy! Select the text above and use Ctrl+C")
                        with col2:
                            if st.button(f"🔗 Generate Link", key=f"generate_link_{i}_{session_key}"):
                                link_id = f"source_{i}_{session_key}"
                                st.code(f"[Source {i}](#{link_id})", language="markdown")
                    
                    with tab2:
                        # Smart citation with multiple formats
                        st.markdown("**📝 Citation formats:**")
                        
                        excerpt = content[:150] + "..." if len(content) > 150 else content
                        citation_formats = {
                            "Academic": f'According to {source} (Page {page}, relevance {relevance:.1f}%): "{excerpt}"',
                            "Brief": f'{source}, p. {page} ({relevance:.1f}%)',
                            "Markdown": f'> {excerpt}\n> \n> *Source: {source}, Page {page}*',
                            "Structured": f'**Source:** {source}\n**Page:** {page}\n**Relevance:** {relevance:.1f}%\n**Content:** "{excerpt}"'
                        }
                        
                        for format_name, citation in citation_formats.items():
                            st.markdown(f"**{format_name} Style:**")
                            st.code(citation, language="text")
                            if st.button(f"📋 Copy {format_name}", key=f"copy_{format_name}_{i}_{session_key}"):
                                st.success(f"✅ {format_name} citation ready to copy!")
                    
                    with tab3:
                        # Key excerpts with similarity matching
                        st.markdown("**🎯 Most relevant excerpts:**")
                        excerpts = _extract_key_excerpts(content, query, max_excerpts=3)
                        
                        for j, excerpt in enumerate(excerpts, 1):
                            similarity = _calculate_similarity(excerpt, query)
                            st.markdown(f"""
                            <div class="passage-highlight" style="margin: 8px 0;">
                                <strong>Excerpt {j}</strong> ({similarity:.1f}% match):<br/>
                                <em>"{excerpt}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab4:
                        # Content analysis
                        st.markdown("**📊 Content Analysis:**")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        with analysis_col1:
                            word_count = len(content.split())
                            sentences = len([s for s in content.split('.') if s.strip()])
                            st.metric("Word Count", word_count)
                            st.metric("Sentences", sentences)
                        
                        with analysis_col2:
                            keywords = _extract_keywords(content, top_k=5)
                            st.markdown("**Key Terms:**")
                            for keyword in keywords:
                                st.markdown(f"- {keyword}")
                
                # Visual separator between passages
                if i < len(passages):
                    st.markdown("---")
        
        # Enhanced summary and next steps
        st.markdown("### ✅ Interactive Source Display Complete")
        
        # Provide guidance based on current state
        if selected_passage:
            st.success(f"🎯 **Passage #{selected_passage} is currently selected** - Use the citation tools above for referencing!")
        else:
            st.info("💡 **Tips:**\n"
                   "- Click on passage headers to select and highlight them\n"
                   "- Use 'Highlight Mode' to enhance visual feedback\n"
                   "- Each passage offers multiple citation formats\n"
                   "- Key excerpts show the most relevant parts of each passage")
        
        # Interactive features summary
        with st.expander("🔧 Interactive Features Guide", expanded=False):
            st.markdown("""
            **🎯 Highlighting Features:**
            - **Click passage headers** to select/deselect passages
            - **Highlight Mode** enhances visual feedback
            - **Keyword highlighting** within content
            - **Relevance bars** show visual importance
            
            **📋 Citation Tools:**
            - **Multiple citation formats** (Academic, Brief, Markdown, Structured)
            - **Quick copy buttons** for easy referencing
            - **Smart excerpts** with similarity matching
            - **Link generation** for cross-referencing
            
            **📊 Analysis Features:**
            - **Content metrics** (word count, sentences)
            - **Key term extraction** from each passage
            - **Relevance scoring** with visual indicators
            - **Similarity matching** for excerpts
            """)
        
        print(f"✅ Successfully displayed {len(passages)} passages with enhanced interactivity")
        return f"✅ Successfully displayed {len(passages)} source passages with interactive highlighting and bidirectional links"
        
    except ImportError:
        error_msg = "❌ Streamlit not available - cannot display enhanced passages in UI"
        print(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"❌ Error displaying enhanced passages in Streamlit: {str(e)}"
        print(error_msg)
        return error_msg


def _highlight_keywords_in_content(content: str, query: str) -> str:
    """Highlight keywords from the query within the content."""
    if not query:
        return content
    
    # Extract keywords from query (remove common stop words)
    stop_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'un', 'une', 'dans', 'pour', 'avec', 'sur'}
    keywords = [word.strip() for word in query.lower().split() if len(word) > 2 and word.lower() not in stop_words]
    
    highlighted_content = content
    for keyword in keywords:
        # Use case-insensitive regex to highlight keywords
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_content = pattern.sub(
            f'<mark style="background-color: #ffeb3b; padding: 1px 3px; border-radius: 3px;">{keyword}</mark>',
            highlighted_content
        )
    
    return highlighted_content


def _extract_key_excerpts(content: str, query: str, max_excerpts: int = 3) -> list:
    """Extract the most relevant excerpts from content based on query."""
    if not content or not query:
        return []
    
    # Split content into sentences
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
    
    # Calculate similarity for each sentence
    scored_sentences = []
    for sentence in sentences:
        similarity = _calculate_similarity(sentence, query)
        scored_sentences.append((sentence, similarity))
    
    # Sort by similarity and return top excerpts
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in scored_sentences[:max_excerpts]]


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100


def _extract_keywords(content: str, top_k: int = 5) -> list:
    """Extract top keywords from content."""
    import re
    from collections import Counter
    
    # Simple keyword extraction (remove stop words and short words)
    stop_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'un', 'une', 'dans', 'pour', 'avec', 'sur', 'par', 'ce', 'qui', 'que', 'est', 'sont', 'ont', 'cette', 'ces'}
    
    # Extract words (alphabetic only, length > 3)
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', content.lower())
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequency and return top k
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_k)]


# Test function for display_source_passages (Streamlit version)
def test_display_source_passages_streamlit():
    """Test the Streamlit display functionality."""
    
    print("🧪 Testing display_source_passages with Streamlit...")
    
    # Mock search results from the logs you provided
    test_results = {
        "success": True,
        "results": [
            {
                "rank": 1,
                "content": "## 2. La gouvernance du dispositif de LCB-FT des groupes\n\n### 2.1 L'organisation du dispositif de LCB-FT",
                "metadata": {
                    "source": "Lignes directrices relatives au pilotage consolidé du dispositif de LCB-FT des groupes.pdf",
                    "page": 5,
                    "relevance_score": 3.9,
                    "content_length": 104
                }
            },
            {
                "rank": 2,
                "content": "19. En outre, conformément aux exigences de l'article L. 511-34, les entreprises mères établissent des procédures permettant l'échange d'informations nécessaires à la LCB-FT.",
                "metadata": {
                    "source": "Lignes directrices relatives au pilotage consolidé du dispositif de LCB-FT des groupes.pdf",
                    "page": 5,
                    "relevance_score": 0.4,
                    "content_length": 150
                }
            }
        ],
        "summary": {
            "total_content_length": 254,
            "average_relevance": 2.15,
            "top_relevance": 3.9
        },
        "context": {
            "filename": "Lignes directrices relatives au pilotage consolidé du dispositif de LCB-FT des groupes.pdf",
            "classification": "legal AML",
            "total_available_files": 1
        }
    }
    
    import json
    test_json = json.dumps(test_results, ensure_ascii=False)
    
    # This would normally display in Streamlit
    result = display_source_passages_enhanced(test_json, "recommandations ACPR gestion de fortune LCB-FT")
    print(f"✅ Test result: {result}")
    return result

# Uncomment to test: test_display_source_passages_streamlit()

@tool
def display_source_passages(search_results: str, query: str = "") -> str:
    """
    Display source passages in Streamlit interface with interactive components.
    
    This is now an alias for the enhanced version with interactive highlighting.
    Maintains backward compatibility while providing improved functionality.
    
    Args:
        search_results: JSON string containing search results with passages and metadata
        query: Original search query for context display
        
    Returns:
        Confirmation message that passages were displayed in Streamlit interface
    """
    # Use the enhanced version by default
    return display_source_passages_enhanced(search_results, query)

# End of RAG tools - All highlighting functionality removed for simplification

@tool
def validate_source_display_before_final_answer(final_answer_text: str) -> str:
    """
    Validation tool that ensures sources have been displayed before allowing final answer.
    
    This tool enforces the mandatory workflow by checking if sources were displayed
    and only allowing final_answer() if the proper workflow was followed.
    
    Args:
        final_answer_text: The text that would be returned as final answer
        
    Returns:
        Either approval to proceed or mandatory workflow enforcement
    """
    
    try:
        import streamlit as st
        
        # Check if sources have been displayed in this session
        sources_displayed = False
        
        # Check various session state keys that indicate sources were displayed
        for key in st.session_state.keys():
            if any(indicator in key.lower() for indicator in ['sources', 'passages', 'search_results', 'highlight']):
                sources_displayed = True
                break
                
        # Also check if display functions were called (basic heuristic)
        if hasattr(st.session_state, 'sources_displayed_flag'):
            sources_displayed = st.session_state.sources_displayed_flag
            
        if not sources_displayed:
            return json.dumps({
                "status": "BLOCKED",
                "error": "🚨 MANDATORY WORKFLOW VIOLATION",
                "message": "You MUST display sources before calling final_answer()",
                "required_action": "Call search_pdf_interactive() or display_source_passages() first",
                "workflow_reminder": """
MANDATORY STEPS:
1. search_pdf_interactive(query) OR
2. smart_pdf_search_for_streamlit(query) + display_source_passages(results, query)
3. THEN final_answer(text)
                """,
                "blocked_text": final_answer_text[:100] + "..."
            })
            
        # If sources were displayed, allow the final answer
        return json.dumps({
            "status": "APPROVED", 
            "message": "✅ Sources displayed. Final answer approved.",
            "final_answer_text": final_answer_text
        })
        
    except ImportError:
        # If not in Streamlit environment, assume validation passes
        return json.dumps({
            "status": "APPROVED",
            "message": "✅ Validation passed (non-Streamlit environment)",
            "final_answer_text": final_answer_text
        })
    except Exception as e:
        return json.dumps({
            "status": "ERROR",
            "error": f"Validation failed: {str(e)}",
            "final_answer_text": final_answer_text
        })


@tool
def unified_pdf_search_and_analyze(query: str) -> str:
    """
    Unified tool that searches PDFs and provides analysis with sources in one step.
    
    Following smolagents best practice: "group 2 tools in one" to reduce LLM calls.
    This tool combines search + display + basic analysis in a single operation.
    
    Args:
        query: The search query for PDF documents
        
    Returns:
        Complete response with sources and analysis combined
    """
    
    if not query or not isinstance(query, str):
        return "❌ Veuillez fournir une requête de recherche valide."
    
    print(f"🔍 Recherche unifiée PDF pour : '{query}'")
    
    try:
        # Step 1: Search PDFs using existing smart search
        search_results = smart_pdf_search_for_streamlit(query)
        
        # Step 2: Parse and display results
        import json as json_module
        try:
            results_data = json_module.loads(search_results)
        except:
            return f"❌ Erreur lors de l'analyse des résultats de recherche."
            
        if not results_data.get("success", False):
            return f"❌ Recherche échouée : {results_data.get('error', 'Erreur inconnue')}"
        
        # Step 3: Display sources interactively
        display_result = search_pdf_interactive(query)
        
        # Step 4: Provide structured response
        passages = results_data.get("passages", [])
        if not passages:
            return "❌ Aucun document pertinent trouvé pour cette requête."
        
        # Build response with sources and analysis
        response_parts = [
            f"✅ **Résultats de recherche pour : '{query}'**",
            "",
            "📄 **Sources trouvées :**"
        ]
        
        for i, passage in enumerate(passages[:3], 1):  # Limit to top 3
            filename = passage.get("filename", "Document inconnu")
            page = passage.get("page", "?")
            relevance = passage.get("relevance_score", 0)
            content = passage.get("content", "")[:200] + "..."
            
            response_parts.extend([
                f"{i}. **{filename}** (Page {page}, Pertinence: {relevance:.1%})",
                f"   *Extrait:* {content}",
                ""
            ])
        
        response_parts.extend([
            "💡 **Analyse rapide :**",
            f"J'ai trouvé {len(passages)} passage(s) pertinent(s) dans les documents PDF.",
            "Les sources sont affichées de manière interactive ci-dessus pour une exploration détaillée.",
            "",
            "ℹ️  Utilisez l'interface interactive ci-dessus pour explorer les résultats complets avec citations et filtres."
        ])
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"❌ Erreur lors de la recherche PDF unifiée : {str(e)}"
