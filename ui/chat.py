"""
Chat interface functionality for handling user interactions.
"""
import streamlit as st
import time
import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
from agents.agent_manager_multiagent import MultiAgentManager
from .rag_display import display_structured_rag_response, display_notebooklm_response
from agents.tools.rag_tools import get_last_rag_sources, _clear_session_sources
import re


def display_chat_interface(model, agent_manager):
    """Display the chat interface and handle user interactions."""
    # Display existing messages with enhanced formatting
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if this is a system/notification message
            is_system_message = _is_system_message(message["content"])
            
            if message["role"] == "assistant":
                if is_system_message:
                    # Special formatting for system messages (file uploads, indexing, etc.)
                    _display_system_message(message["content"], message.get("timestamp"))
                else:
                    # Try to display as interactive RAG results, fallback to text
                    if not display_structured_rag_response(message["content"], ""):
                        # Regular assistant response
                        st.markdown(message["content"])
                    _display_timestamp(message.get("timestamp"))
            else:
                # User message
                st.markdown(message["content"])
                _display_timestamp(message.get("timestamp"))

    # Handle new user input
    if prompt := st.chat_input("Quel est votre question?"):
        # Add timestamp to user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        }
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)
            _display_timestamp(user_message["timestamp"])

        with st.chat_message("assistant"):
            # Create a container for the response that can be updated
            response_container = st.container()
            
            with response_container:
                status_placeholder = st.empty()
                status_placeholder.markdown("🔄 **Traitement en cours...**")
                
                # Process the query and get response
                final_response = _process_user_query(
                    prompt, model, agent_manager,
                    response_container  # Pass container for real-time updates
                )
                
                # Clear the status and show final response
                status_placeholder.empty()
                
                # Try to display as interactive RAG results first
                if display_structured_rag_response(final_response, prompt):
                    # Interactive display was successful
                    # Still store the response but format it for chat history
                    formatted_response = _format_agent_response_for_history(final_response)
                else:
                    # Fallback to regular text display
                    # Affichage direct de la réponse brute
                    formatted_response = str(final_response)
                    st.markdown(formatted_response)
                
                # Store the formatted response in session state with timestamp
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": formatted_response,
                    "timestamp": time.time()
                })


def _is_system_message(content):
    """Check if a message is a system/notification message."""
    system_indicators = [
        "📤", "✅", "🔄", "❌", "⚠️", "ℹ️", "🚀", "🎉", "💥",
        "**Début du traitement**", "**traité avec succès**", "**Démarrage de l'indexation**",
        "**Indexation terminée**", "**Configuration en cours**", "**Erreur"
    ]
    return any(indicator in content for indicator in system_indicators)


def _display_system_message(content, timestamp=None):
    """Display system messages with special formatting."""
    # Create a bordered container for system messages
    with st.container():
        # Use info/success/error styling based on content
        if "✅" in content or "🎉" in content or "traité avec succès" in content:
            st.success(content)
        elif "❌" in content or "💥" in content or "Erreur" in content:
            st.error(content)
        elif "⚠️" in content:
            st.warning(content)
        elif "🔄" in content or "en cours" in content:
            st.info(content)
        else:
            st.info(content)
    
    _display_timestamp(timestamp)


def _display_timestamp(timestamp):
    """Display a timestamp for messages."""
    if timestamp:
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S")
            st.caption(f"🕒 {time_str}")
        except (ValueError, TypeError):
            pass


def _process_user_query(prompt, model, agent_manager, response_container=None):
    """Process the user query and return the response using the new clean architecture."""
    if not model or not agent_manager:
        return "Désolé, les agents IA ne sont pas correctement initialisés. Vérifiez la configuration de la clé API."
    
    try:
        available_pdfs_context, available_csvs_context = _prepare_context()
        
        # Show progress if we have a container
        if response_container:
            with response_container:
                progress_placeholder = st.empty()
                progress_placeholder.markdown("🤖 **Agent en cours d'exécution...**")
        
        # Use the new clean AgentManager interface
        final_response = agent_manager.run_task(
            user_query=prompt,
            additional_args={
                'pdf_context': {'available_files': available_pdfs_context},
                'csv_context': {'available_files': available_csvs_context}
            }
        )
        
        # Clear progress indicator
        if response_container:
            progress_placeholder.empty()
        
        # 🆕 PRIORITÉ 1: Post-traitement avec cache des sources (bypass total de l'agent)
        print(f"🚀 Sources Processing: Starting post-processing pipeline...")
        
        clean_response, sources_displayed = _process_agent_response_with_sources(final_response, prompt)
        
        if sources_displayed:
            print(f"✅ Sources Processing: Sources successfully displayed via cache system")
            return clean_response
        
        print(f"🔄 Sources Processing: No cached sources found, proceeding with fallback detection...")
        
        # Use the cleaned response for further processing
        final_response = clean_response
        
        # Try to display as structured RAG response first
        try:
            if display_structured_rag_response(final_response, prompt):
                return final_response
        except Exception as e:
            print(f"⚠️ DEBUG: Error in display_structured_rag_response fallback: {e}")
        
        # 🆕 FALLBACK: Essayer de détecter manuellement les réponses RAG
        print(f"🔄 DEBUG Fallback: Checking for RAG patterns manually...")
        
        # Détecter les citations [1], [2], etc.
        citations_found = bool(re.search(r'\[\d+\]', final_response))
        # Détecter les blocs JSON
        json_found = bool(re.search(r'```json\s*\n.*?\n```', final_response, re.DOTALL))
        
        print(f"🔍 DEBUG Fallback: Citations found: {citations_found}")
        print(f"🔍 DEBUG Fallback: JSON block found: {json_found}")
        
        if citations_found and json_found:
            print(f"🎯 DEBUG Fallback: Manual RAG detection successful!")
            try:
                if display_structured_rag_response(final_response, prompt):
                    print(f"✅ DEBUG Fallback: Successfully displayed via manual detection")
                    return final_response
                else:
                    print(f"⚠️ DEBUG Fallback: display_structured_rag_response returned False")
            except Exception as e:
                print(f"❌ DEBUG Fallback: Error in manual RAG display: {e}")
        
        return final_response
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement de la requête: {str(e)}"
        print(f"❌ Error in _process_user_query: {error_msg}")
        return error_msg


def _format_agent_response_for_history(response):
    """Format agent response for chat history when interactive display is used."""
    if not response:
        return "Aucune réponse générée."
    
    # For RAG responses with interactive display, store a clean summary
    formatted_response = str(response)
    
    # Remove the streamlit display markers from history
    if "```streamlit_rag_display" in formatted_response:
        lines = formatted_response.split('\n')
        clean_lines = []
        skip_block = False
        
        for line in lines:
            if line.strip().startswith("```streamlit_rag_display"):
                skip_block = True
                continue
            elif skip_block and line.strip() == "```":
                skip_block = False
                continue
            elif not skip_block:
                clean_lines.append(line)
        
        formatted_response = '\n'.join(clean_lines).strip()
    
    return formatted_response if formatted_response else "Résultats affichés ci-dessus."


def _prepare_context():
    """Prepare PDF and CSV context for the manager agent."""
    available_pdfs_context = []
    available_csvs_context = []
    
    for fid, details in st.session_state.get('processed_files', {}).items():
        if details.get('type') == 'pdf' and details.get('indexed') and details.get('db_path'):
            available_pdfs_context.append({
                'file_id': fid,
                'filename': details.get('filename', 'Unknown PDF'),
                'classification': details.get('classification'),
                'db_path': details.get('db_path'),
                'user_notes': details.get('user_notes', ''),
                'summary': details.get('summary', '')
            })
        elif details.get('type') == 'csv' and details.get('status') == 'ready':
            csv_args = details.get('csv_args', {})
            if not isinstance(csv_args, dict): 
                csv_args = {}
            
            csv_context = {
                'file_id': fid,
                'filename': details.get('filename', 'Unknown CSV'),
                'csv_args': csv_args,
                'user_notes': details.get('user_notes', '')
            }
            
            if 'rows' in details:
                csv_context['csv_args']['rows'] = details['rows']
            if 'columns' in details:
                csv_context['csv_args']['columns'] = details['columns']
                
            available_csvs_context.append(csv_context)
    
    return available_pdfs_context, available_csvs_context 


def _process_agent_response_with_sources(agent_response: str, query: str = "") -> tuple[str, bool]:
    """
    Post-traitement des réponses d'agent pour extraire et afficher les sources de la session.
    
    Args:
        agent_response: Réponse brute de l'agent
        query: Requête originale de l'utilisateur
        
    Returns:
        Tuple (response_clean, sources_displayed)
    """
    print(f"🔍 Sources Post-Processing: Checking for sources in session...")
    
    # Récupérer les sources de la session
    sources_info = get_last_rag_sources()
    
    if sources_info:
        sources_data = sources_info.get("sources_data")
        source_query = sources_info.get("query", "")
        
        print(f"📦 Sources Post-Processing: Found sources in session for query: '{source_query[:50]}...'")
        
        if sources_data:
            print(f"✅ Sources Post-Processing: Sources loaded successfully")
            
            try:
                # Afficher via l'interface NotebookLM
                display_notebooklm_response(agent_response, sources_data, query)
                print(f"🎨 Sources Post-Processing: NotebookLM display successful")
                
                # Nettoyer la session
                _clear_session_sources()
                
                return agent_response, True
                
            except Exception as e:
                print(f"❌ Sources Post-Processing: Display error: {e}")
                # Nettoyer la session même en cas d'erreur
                _clear_session_sources()
        else:
            print(f"⚠️ Sources Post-Processing: No sources data found in session")
    else:
        print(f"ℹ️ Sources Post-Processing: No sources found in session")
    
    return agent_response, False 