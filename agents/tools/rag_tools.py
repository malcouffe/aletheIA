"""
RAG Tools for Document Retrieval Agents - Enhanced with Debug
Contains only the unified PDF search tool following smolagents best practices with comprehensive debugging.

Note: This module uses print() statements for RAG debugging while HTTP/API logs are suppressed.
The print() statements with "🔍 RAG Debug:" prefixes show the RAG pipeline execution flow.
"""

import os
import json
import logging
import re
import time
import streamlit as st
from smolagents import tool
from ..config.agent_config import RAG_CONFIG

# Configure logging for RAG tools debug - disable HTTP/API logs
logging.basicConfig(level=logging.WARNING)  # Set global level to WARNING
logger = logging.getLogger(__name__)

# Disable verbose HTTP/API logs
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def _search_pdfs_internal(user_query: str) -> str:
    """
    Internal function to search PDFs and return structured results.
    Enhanced with comprehensive debug logging following smolagents best practices.
    """
    print(f"🔍 RAG Debug: Starting PDF search for query: '{user_query}'")
    
    if not user_query or not isinstance(user_query, str):
        error_msg = "Please provide a valid search query"
        print(f"❌ RAG Debug: Validation error - {error_msg}")
        return json.dumps({
            "success": False,
            "error": error_msg,
            "error_type": "validation_error",
            "debug_info": {
                "query_type": type(user_query).__name__,
                "query_length": len(user_query) if user_query else 0
            }
        })
    
    print(f"✅ RAG Debug: Query validation passed - length: {len(user_query)} chars")
    
    # Try to get context from session state
    try:
        import json as json_module
        
        session_file = "data/session_persistence/persistent_session_state.json"
        print(f"🔍 RAG Debug: Checking session file: {session_file}")
        
        if os.path.exists(session_file):
            print(f"✅ RAG Debug: Session file found, loading...")
            with open(session_file, 'r', encoding='utf-8') as f:
                processed_files = json_module.load(f)
            
            print(f"📊 RAG Debug: Loaded {len(processed_files)} files from session")
            
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
                    print(f"📄 RAG Debug: Found indexed PDF - {details.get('filename')} at {details.get('db_path')}")
            
            print(f"📊 RAG Debug: Total indexed PDFs found: {len(available_pdfs_context)}")
            
            if available_pdfs_context:
                # Use first available PDF for search
                first_pdf = available_pdfs_context[0]
                db_path = first_pdf.get('db_path')
                filename = first_pdf.get('filename', '')
                
                print(f"🎯 RAG Debug: Using PDF '{filename}' for search")
                print(f"🗂️ RAG Debug: Database path: {db_path}")
                
                if db_path and os.path.exists(db_path):
                    print(f"✅ RAG Debug: Database path exists, proceeding with vector search")
                    return _perform_vector_search(user_query, db_path, filename)
                else:
                    print(f"❌ RAG Debug: Database path does not exist: {db_path}")
            else:
                print(f"⚠️ RAG Debug: No indexed PDFs found in session")
        else:
            print(f"⚠️ RAG Debug: Session file does not exist: {session_file}")
        
    except Exception as e:
        print(f"❌ RAG Debug: Error loading session context: {str(e)}")
        print(f"🔧 RAG Debug: Exception type: {type(e).__name__}")
        logger.exception("Session context loading failed")
    
    # No context found - return guidance
    print(f"📖 RAG Debug: No context found, returning guidance message")
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
        "query": user_query,
        "debug_info": {
            "session_file_checked": session_file,
            "session_file_exists": os.path.exists(session_file) if 'session_file' in locals() else False
        }
    }, ensure_ascii=False, indent=2)


def _perform_vector_search(query: str, db_path: str, filename: str) -> str:
    """
    Perform the actual vector search in the PDF database.
    Enhanced with detailed debug logging and error handling.
    """
    print(f"🔍 RAG Debug: Starting vector search")
    print(f"🔍 RAG Debug: Query: '{query}'")
    print(f"🔍 RAG Debug: DB Path: '{db_path}'")
    print(f"🔍 RAG Debug: Filename: '{filename}'")
    
    try:
        print(f"📚 RAG Debug: Importing vector store dependencies...")
        from langchain_community.vectorstores import Chroma
        from ..core.embedding import get_embedding_function
        print(f"✅ RAG Debug: Dependencies imported successfully")
        
        print(f"🔧 RAG Debug: Getting embedding function...")
        embedding_function = get_embedding_function()
        print(f"✅ RAG Debug: Embedding function obtained: {type(embedding_function).__name__}")
        
        # Initialize the Chroma vector store
        print(f"🗂️ RAG Debug: Initializing Chroma vector store...")
        print(f"🗂️ RAG Debug: Collection name: {RAG_CONFIG['collection_name']}")
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=RAG_CONFIG["collection_name"]
        )
        print(f"✅ RAG Debug: Vector store initialized successfully")
        
        # Perform similarity search
        search_k = RAG_CONFIG["similarity_search_k"]
        print(f"🎯 RAG Debug: Performing similarity search with k={search_k}")
        
        results = vector_store.similarity_search_with_score(query, k=search_k)
        print(f"📊 RAG Debug: Search completed - found {len(results)} results")
        
        if not results:
            print(f"⚠️ RAG Debug: No results found for query")
            return json.dumps({
                "success": True,
                "results": [],
                "query": query,
                "context": {"filename": filename},
                "message": "Aucun document pertinent trouvé pour cette requête",
                "debug_info": {
                    "search_k": search_k,
                    "db_path": db_path,
                    "collection_name": RAG_CONFIG["collection_name"]
                }
            })

        # Structure results
        print(f"🔧 RAG Debug: Processing {len(results)} search results...")
        structured_results = []
        total_content_length = 0
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata or {}
            content = doc.page_content.strip()
            content_length = len(content)
            total_content_length += content_length
            
            # Calculate relevance percentage
            relevance_percent = (1 - score) * 100 if score <= 1 else 0
            
            # FIX: Le problème principal - utiliser la bonne clé pour les métadonnées
            source_name = metadata.get('source') or metadata.get('filename') or filename
            page_number = metadata.get('page', 'Page inconnue')
            
            print(f"📄 RAG Debug: Result {i} - Score: {score:.4f}, Relevance: {relevance_percent:.1f}%, Length: {content_length}")
            print(f"📄 RAG Debug: Source: {source_name}, Page: {page_number}")
            
            result_item = {
                "rank": i,
                "content": content,
                "metadata": {
                    "source": source_name,  # Utilise le source_name corrigé
                    "page": page_number,
                    "relevance_score": round(relevance_percent, 1),
                    "content_length": content_length,
                    "raw_score": score  # Add raw score for debugging
                }
            }
            structured_results.append(result_item)
        
        # Calculate average relevance
        avg_relevance = sum(r["metadata"]["relevance_score"] for r in structured_results) / len(structured_results)
        
        print(f"📊 RAG Debug: Search summary:")
        print(f"📊 RAG Debug: - Total content length: {total_content_length}")
        print(f"📊 RAG Debug: - Average relevance: {avg_relevance:.1f}%")
        print(f"📊 RAG Debug: - Top relevance: {structured_results[0]['metadata']['relevance_score']:.1f}%")
        
        return json.dumps({
            "success": True,
            "results": structured_results,
            "query": query,
            "context": {"filename": filename},
            "summary": {
                "total_content_length": total_content_length,
                "average_relevance": round(avg_relevance, 1),
                "top_relevance": structured_results[0]["metadata"]["relevance_score"] if structured_results else 0
            },
            "debug_info": {
                "search_k": search_k,
                "db_path": db_path,
                "collection_name": RAG_CONFIG["collection_name"],
                "results_count": len(structured_results),
                "embedding_function": type(embedding_function).__name__
            }
        }, ensure_ascii=False, indent=2)

    except ImportError as e:
        error_msg = f"Missing dependencies: {str(e)}"
        print(f"❌ RAG Debug: Import error - {error_msg}")
        print(f"💡 RAG Debug: Required packages: langchain-community, chromadb")
        return json.dumps({
            "success": False,
            "error": error_msg,
            "error_type": "import_error",
            "debug_info": {
                "missing_dependencies": ["langchain-community", "chromadb"],
                "suggestion": "Install with: pip install langchain-community chromadb"
            }
        })
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        print(f"❌ RAG Debug: Search error - {error_msg}")
        print(f"🔧 RAG Debug: Exception type: {type(e).__name__}")
        logger.exception("Vector search failed")
        return json.dumps({
            "success": False,
            "error": error_msg,
            "error_type": "search_error",
            "debug_info": {
                "query": query,
                "db_path": db_path,
                "filename": filename,
                "exception_type": type(e).__name__
            }
        })


def _generate_notebooklm_response(results: dict, user_query: str) -> str:
    """
    Génère une réponse style NotebookLM avec citations numérotées intégrées.
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
def unified_pdf_search_and_analyze(query: str) -> str:
    """
    Unified tool that searches PDFs and provides analysis with sources in one step.
    
    Following smolagents best practice: "group 2 tools in one" to reduce LLM calls.
    This tool combines search + display + basic analysis in a single operation.
    Enhanced with comprehensive debug logging and error handling.
    
    Args:
        query: The search query for PDF documents
        
    Returns:
        Complete response with sources and analysis combined in NotebookLM format
    """
    
    print(f"🚀 RAG Tool: Starting unified PDF search and analysis")
    print(f"🔍 RAG Tool: Query received: '{query}'")
    
    if not query or not isinstance(query, str):
        error_msg = "❌ RAG Tool: Query must be a non-empty string"
        print(error_msg)
        return error_msg
    
    print(f"✅ RAG Tool: Query validation passed")
    
    try:
        print(f"🔍 RAG Tool: Unified PDF search and analysis for: '{query}'")
        print(f"📊 RAG Tool: Calling internal search function...")
        
        # Call the internal search function
        search_results = _search_pdfs_internal(query)
        
        print(f"📊 RAG Tool: Internal search completed")
        print(f"🔧 RAG Tool: Parsing search results...")
        
        # Parse search results
        try:
            results_data = json.loads(search_results)
            print(f"✅ RAG Tool: Results parsed successfully")
        except json.JSONDecodeError as e:
            error_msg = f"❌ RAG Tool: Failed to parse search results: {e}"
            print(error_msg)
            return error_msg
        
        # Check if search was successful
        if not results_data.get("success", False):
            error_msg = f"❌ RAG Tool: Search failed: {results_data.get('error', 'Unknown error')}"
            print(error_msg)
            return error_msg
        
        print(f"✅ RAG Tool: Search successful, generating NotebookLM response...")
        
        # Get search results and query for response generation
        search_results_list = results_data.get("results", [])
        print(f"📊 RAG Tool: Found {len(search_results_list)} results")
        
        if not search_results_list:
            no_results_msg = f"ℹ️ Aucun document trouvé pour la requête '{query}'. Essayez avec des termes différents."
            print(f"📊 RAG Tool: {no_results_msg}")
            return no_results_msg
        
        print(f"🎨 RAG Tool: Generating NotebookLM-style response...")
        
        # Generate the comprehensive response
        notebooklm_response = _generate_notebooklm_response(results_data, query)
        
        print(f"✅ RAG Tool: NotebookLM response generated")
        print(f"🔍 RAG Tool: NotebookLM response preview: {notebooklm_response[:200]}...")
        
        # Combine with JSON metadata for complete response
        print(f"🔗 RAG Tool: Combining response with JSON metadata...")
        
        # Create the combined response with sources
        combined_response = f"{notebooklm_response}\n\n```json\n{search_results}\n```"
        
        print(f"✅ RAG Tool: Tool execution completed successfully")
        print(f"📊 RAG Tool: Final response length: {len(combined_response)} characters")
        
        # 🆕 NOUVELLE APPROCHE: Cache des sources hors-LLM
        # Vérifier si la réponse contient des citations (format NotebookLM)
        has_citations = bool(re.search(r'\[\d+\]', notebooklm_response))
        has_sources = len(search_results_list) > 0
        
        print(f"🔍 RAG Tool: Citation check - Has citations: {has_citations}")
        print(f"🔍 RAG Tool: Sources check - Has sources: {has_sources}")
        
        if has_citations and has_sources:
            print(f"🎯 RAG Tool: Detected structured response with citations and sources")
            
            # Sauvegarder les sources dans la session Streamlit
            _store_sources_in_session(results_data, query)
            
            print(f"📦 RAG Tool: Sources stored in session, returning clean response")
            # Retourner seulement la réponse naturelle sans cache ID
            return notebooklm_response
        else:
            print(f"ℹ️ RAG Tool: Response doesn't have citations/sources, returning normally")
            return combined_response
            
    except Exception as e:
        error_msg = f"❌ RAG Tool: Unexpected error during execution: {str(e)}"
        print(error_msg)
        import traceback
        print(f"📍 RAG Tool: Traceback: {traceback.format_exc()}")
        return error_msg


def _store_sources_in_session(sources_data: dict, query: str):
    """
    Stocke les sources dans la session Streamlit pour affichage automatique.
    
    Args:
        sources_data: Données JSON des sources
        query: Requête originale
    """
    try:
        if 'last_rag_sources' not in st.session_state:
            st.session_state.last_rag_sources = {}
        
        st.session_state.last_rag_sources = {
            "query": query,
            "sources_data": sources_data,
            "timestamp": time.time()
        }
        
        print(f"📦 RAG Session: Sources stored in session for query: '{query[:50]}...'")
        
    except Exception as e:
        print(f"❌ RAG Session: Failed to store sources in session: {e}")


def _clear_session_sources():
    """Nettoie les sources de la session."""
    try:
        if 'last_rag_sources' in st.session_state:
            del st.session_state.last_rag_sources
            print(f"🧹 RAG Session: Cleared sources from session")
    except Exception as e:
        print(f"⚠️ RAG Session: Failed to clear sources: {e}")


def get_last_rag_sources():
    """
    Récupère les dernières sources RAG de la session.
    
    Returns:
        Dict contenant les données sources ou None
    """
    try:
        return st.session_state.get('last_rag_sources', None)
    except Exception as e:
        print(f"❌ RAG Session: Failed to get sources from session: {e}")
        return None
