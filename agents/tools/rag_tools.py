"""
RAG Tools for Document Retrieval Agents - Version Simple avec Affichage Direct
Affiche automatiquement les sources trouvées lors de la recherche vectorielle.
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

# Cache temporaire global pour les sources RAG
_temp_rag_sources_cache = []

@tool
def rag_search_simple(query: str) -> str:
    """
    Search PDF documents and return answer with citations.
    AFFICHE AUTOMATIQUEMENT les sources trouvées avec scores de pertinence.
    
    Args:
        query: Search terms for PDF documents.
    
    Returns:
        Answer with citations [SOURCE-1], [SOURCE-2], etc.
    """
    
    print(f"📚 RAG SEARCH: Starting search for: '{query}'")
    
    if not query or not isinstance(query, str):
        return "❌ Query must be a non-empty string"
    
    try:
        # Try to import session state (will fail outside Streamlit)
        try:
            import streamlit as st
            in_streamlit = True
        except:
            in_streamlit = False
            
        # Load session data
        session_file = "data/session_persistence/persistent_session_state.json"
        print(f"📚 RAG SEARCH: Checking session file: {session_file}")
        
        if not os.path.exists(session_file):
            print("❌ RAG SEARCH: No session file found")
            return "❌ Aucun document PDF n'est disponible pour la recherche. Veuillez d'abord télécharger des documents PDF."
        
        print(f"📚 RAG SEARCH: Loading session data...")
        with open(session_file, 'r', encoding='utf-8') as f:
            processed_files = json.load(f)
        
        print(f"📚 RAG SEARCH: Loaded {len(processed_files)} files from session")
        
        # Find indexed PDFs
        available_pdfs = []
        for fid, details in processed_files.items():
            if details.get('type') == 'pdf' and details.get('indexed') and details.get('db_path'):
                available_pdfs.append({
                    'file_id': fid,
                    'filename': details.get('filename', 'Unknown PDF'),
                    'db_path': details.get('db_path'),
                })
                print(f"📄 RAG SEARCH: Found indexed PDF - {details.get('filename')} at {details.get('db_path')}")
        
        if not available_pdfs:
            print("❌ RAG SEARCH: No indexed PDFs found")
            return "❌ Aucun document PDF indexé n'est disponible pour la recherche. Veuillez d'abord indexer des documents PDF."
        
        # Use first available PDF
        first_pdf = available_pdfs[0]
        db_path = first_pdf['db_path']
        filename = first_pdf['filename']
        
        print(f"📚 RAG SEARCH: Using PDF '{filename}' with database: {db_path}")
        
        if not os.path.exists(db_path):
            print(f"❌ RAG SEARCH: Database path does not exist: {db_path}")
            return f"❌ Base de données vectorielle introuvable: {db_path}"
        
        # Initialize vector store
        from langchain_community.vectorstores import Chroma
        from ..core.embedding import get_embedding_function
        
        print(f"📚 RAG SEARCH: Initializing vector store...")
        embedding_function = get_embedding_function()
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=RAG_CONFIG["collection_name"]
        )
        
        # Check if we have documents
        try:
            doc_count = vector_store._collection.count()
            if doc_count == 0:
                print("❌ RAG SEARCH: No documents found in vector store")
                return "❌ Aucun document trouvé dans la base vectorielle."
        except:
            print("❌ RAG SEARCH: Could not access vector store collection")
            return "❌ Erreur d'accès à la base de données vectorielle."
        
        print(f"📚 RAG SEARCH: Found {doc_count} documents in vector store")
        
        # Perform search
        results = vector_store.similarity_search_with_score(query, k=5)
        print(f"📚 RAG SEARCH: Found {len(results)} relevant passages")
        
        if not results:
            print("❌ RAG SEARCH: No relevant passages found")
            return f"❌ Aucun passage pertinent trouvé pour la requête: '{query}'"
        
        # AFFICHAGE DIRECT DÉSACTIVÉ - Les sources seront affichées après la réponse
        print("🎨 RAG SEARCH: Sources seront affichées via le système de cache après la réponse")
        
        # Format results for response
        response_parts = []
        sources_data = []
        
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            page = metadata.get('page', '?')
            source = metadata.get('source') or metadata.get('filename', filename)
            
            # LOGGING: Print detailed info for the LLM agent
            print(f"📄 SOURCE {i}: {source}, page {page} (score: {score:.3f})")
            print(f"📝 CONTENT {i}: {content[:200]}...")
            
            # Add to response with citation
            response_parts.append(f"• {content} [SOURCE-{i}]")
            
            # Store for potential future use
            sources_data.append({
                "content": content,
                "metadata": {
                    "source": source,
                    "page": page,
                    "relevance_score": float(1 - score),  # Convert distance to relevance
                    "citation_index": i
                }
            })
        
        # Build final response with clear page references
        response = f"RÉPONSE DOCUMENTÉE basée sur {len(sources_data)} sources dans le document '{filename}':\n\n"
        
        # Format each passage with clear page references
        formatted_parts = []
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            page = metadata.get('page', '?')
            
            # Format with clear page indication
            formatted_parts.append(f"📄 **Page {page}**: {content} [SOURCE-{i}]")
        
        response += "\n\n".join(formatted_parts)
        response += f"\n\n📚 **RÉFÉRENCES COMPLÈTES:**\n"
        for i, source_data in enumerate(sources_data, 1):
            metadata = source_data["metadata"]
            response += f"[SOURCE-{i}] {metadata['source']}, page {metadata['page']} (pertinence: {metadata['relevance_score']:.0%})\n"
        
        print(f"✅ RAG SEARCH: Successfully generated response with {len(sources_data)} citations")
        print(f"✅ RAG SEARCH: Sources displayed directly in UI")
        
        # Sauvegarder les sources pour affichage automatique
        _save_rag_sources_to_temp_cache(results, query, filename)
        
        return response
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de la recherche RAG: {str(e)}"
        print(f"❌ RAG SEARCH ERROR: {error_msg}")
        return error_msg


def _save_rag_sources_to_temp_cache(sources_data: list, query: str, filename: str) -> None:
    """
    Sauvegarde les sources RAG dans un cache temporaire global.
    """
    global _temp_rag_sources_cache
    
    try:
        if not sources_data:
            print("⚠️ Aucune source à sauvegarder")
            return
            
        print(f"🔄 Tentative de sauvegarde de {len(sources_data)} sources...")
        
        # Préparer les données pour l'affichage
        formatted_sources = {
            "query": query,
            "filename": filename,
            "timestamp": time.time(),
            "results": []
        }
        
        for i, (doc, score) in enumerate(sources_data, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            page = metadata.get('page', '?')
            source = metadata.get('source') or metadata.get('filename', filename)
            relevance = float(1 - score) * 100  # Convert distance to percentage
            
            formatted_sources["results"].append({
                "content": content,
                "metadata": {
                    "source": source,
                    "page": str(page),
                    "relevance_score": relevance,
                    "citation_index": i
                }
            })
        
        # Sauvegarder dans le cache temporaire global
        _temp_rag_sources_cache.append(formatted_sources)
        
        # Garder seulement les 5 dernières recherches
        if len(_temp_rag_sources_cache) > 5:
            _temp_rag_sources_cache = _temp_rag_sources_cache[-5:]
        
        print(f"✅ Sources sauvegardées dans cache temporaire: {len(formatted_sources['results'])} sources pour '{query}'")
        
        # Essayer aussi de sauvegarder dans le session state si possible
        try:
            if 'st' in globals() and hasattr(st, 'session_state'):
                if 'rag_sources_cache' not in st.session_state:
                    st.session_state.rag_sources_cache = []
                st.session_state.rag_sources_cache.append(formatted_sources)
                if len(st.session_state.rag_sources_cache) > 5:
                    st.session_state.rag_sources_cache = st.session_state.rag_sources_cache[-5:]
                print("✅ Sources aussi sauvegardées dans session state")
        except Exception as e:
            print(f"⚠️ Session state indisponible: {e}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde des sources: {e}")
        import traceback
        traceback.print_exc()


def get_latest_rag_sources():
    """
    Récupère les dernières sources RAG sauvegardées (session state ou cache temporaire).
    """
    try:
        # Essayer d'abord le session state
        if 'st' in globals() and hasattr(st, 'session_state') and 'rag_sources_cache' in st.session_state:
            if st.session_state.rag_sources_cache:
                print("📚 Sources récupérées depuis session state")
                return st.session_state.rag_sources_cache[-1]
    except:
        pass
    
    # Fallback vers le cache temporaire global
    try:
        global _temp_rag_sources_cache
        if _temp_rag_sources_cache:
            print("📚 Sources récupérées depuis cache temporaire")
            return _temp_rag_sources_cache[-1]
    except:
        pass
    
    print("⚠️ Aucune source trouvée dans les caches")
    return None


def clear_rag_sources_cache():
    """
    Vide les caches des sources RAG.
    """
    global _temp_rag_sources_cache
    
    try:
        # Vider le cache temporaire
        _temp_rag_sources_cache = []
        
        # Vider le session state si possible
        if 'st' in globals() and hasattr(st, 'session_state') and 'rag_sources_cache' in st.session_state:
            st.session_state.rag_sources_cache = []
        
        print("🧹 Caches des sources RAG vidés")
    except Exception as e:
        print(f"⚠️ Erreur lors du nettoyage des caches: {e}")
