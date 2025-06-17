"""
Chat interface functionality for handling user interactions.
"""
import streamlit as st
import time
import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
from agents.agent_manager_multiagent import SimplifiedMultiAgentManager
from .rag_display import display_notebooklm_response

import re
import json


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
                    # Regular assistant response (système simplifié)
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
                final_response, sources_displayed = _process_user_query(
                    prompt, model, agent_manager,
                    response_container  # Pass container for real-time updates
                )
                
                # Clear the status and show final response
                status_placeholder.empty()
                
                # Affichage simplifié : la réponse et les sources sont gérées dans _process_user_query
                response_for_history = str(final_response)
                
                # Store the formatted response in session state with timestamp
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_for_history,
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


def _rag_step_callback(memory_step, agent):
    """
    Step callback pour gérer l'affichage des sources RAG selon les bonnes pratiques smolagents.
    
    Cette fonction est appelée après chaque étape de l'agent et peut accéder:
    - memory_step: l'étape qui vient d'être exécutée
    - agent: l'agent complet avec sa mémoire
    """
    try:
        # Vérifier si cette étape contenait un appel à l'outil RAG
        if hasattr(memory_step, 'observations_logs') and memory_step.observations_logs:
            logs = str(memory_step.observations_logs)
            
            # Détecter si l'outil RAG a été utilisé
            if "📚 RAG SEARCH:" in logs and "✅ RAG SEARCH: Successfully generated response" in logs:
                print("🔄 STEP CALLBACK: RAG tool detected, sources should be available in session")
                
                # Les sources ont été stockées par l'outil RAG lui-même
                # L'interface les récupérera automatiquement via get_last_rag_sources()
                
    except Exception as e:
        print(f"❌ Step callback error: {e}")


def _process_user_query(prompt, model, agent_manager, response_container=None):
    """Process the user query and return the response - VERSION SIMPLIFIÉE."""
    if not model or not agent_manager:
        return "Désolé, les agents IA ne sont pas correctement initialisés. Vérifiez la configuration de la clé API.", False
    
    try:
        available_pdfs_context, available_csvs_context = _prepare_context()
        
        # Show progress if we have a container
        if response_container:
            with response_container:
                progress_placeholder = st.empty()
                progress_placeholder.markdown("🤖 **Agent en cours d'exécution...**")
        
        # Process query with simplified callbacks
        final_response = agent_manager.process_query(
            prompt=prompt,
            model=model,
            available_pdfs_context=available_pdfs_context,
            available_csvs_context=available_csvs_context,
            step_callbacks=[]  # Pas de callbacks complexes
        )
        
        # Clear progress
        if response_container:
            with response_container:
                progress_placeholder.empty()
        
        # Simple validation
        if not final_response or len(str(final_response).strip()) < 5:
            return "Désolé, je n'ai pas pu générer une réponse satisfaisante.", False
        
        print(f"🎯 Final response generated: {len(str(final_response))} characters")
        
        # Afficher la réponse
        response_text = str(final_response)
        st.markdown(response_text)
        
        # Vérifier si c'est une réponse RAG et afficher les sources sauvegardées
        if _is_rag_response(response_text):
            print("📚 Réponse RAG détectée, affichage des sources sauvegardées...")
            sources_displayed = _display_saved_rag_sources()
            
            if sources_displayed:
                print("✅ Sources sauvegardées affichées avec succès")
                return response_text, True  # True = sources affichées
            else:
                print("⚠️ Aucune source sauvegardée trouvée")
                return response_text, False
        else:
            print("ℹ️ Réponse non-RAG, pas d'affichage de sources")
            return response_text, False
        
    except Exception as e:
        print(f"❌ Error in _process_user_query: {e}")
        import traceback
        traceback.print_exc()
        return f"Erreur lors du traitement: {str(e)}", False


def _clean_response_from_references(response: str) -> str:
    """Nettoie la réponse en enlevant la section des références documentaires."""
    if not response:
        return response
    
    # Couper à l'ancienne section des références documentaires
    if "RÉFÉRENCES DOCUMENTAIRES:" in response:
        parts = response.split("RÉFÉRENCES DOCUMENTAIRES:")
        return parts[0].strip()
    
    # Couper à la nouvelle section des références complètes
    if "📚 **RÉFÉRENCES COMPLÈTES:**" in response:
        parts = response.split("📚 **RÉFÉRENCES COMPLÈTES:**")
        return parts[0].strip()
    
    return response


def _highlight_citations_in_response(response: str) -> str:
    """Applique la colorisation des citations dans la réponse."""
    from .rag_display import _highlight_citations
    return _highlight_citations(response)


# Fonction supprimée - affichage désormais direct dans rag_search_simple


def _extract_embedded_sources(response: str) -> dict:
    """Extrait les sources embarquées dans la réponse de l'agent."""
    if not response:
        return None
    
    # Chercher le bloc de sources embarquées
    import re
    pattern = r'```streamlit_rag_display\n(.*?)\n```'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        try:
            sources_json = match.group(1)
            sources_data = json.loads(sources_json)
            print(f"✅ Extracted embedded sources: {len(sources_data.get('results', []))} results")
            return sources_data
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse embedded sources: {e}")
            return None
    
    # Tenter d'extraire les sources du texte de la réponse
    return _extract_sources_from_text(response)


def _extract_sources_from_text(response: str) -> dict:
    """Extrait les sources directement du texte de la réponse."""
    # Vérifier les deux formats possibles
    has_old_format = "RÉFÉRENCES DOCUMENTAIRES" in response
    has_new_format = "📚 **RÉFÉRENCES COMPLÈTES:**" in response
    
    if not response or (not has_old_format and not has_new_format):
        return None
    
    try:
        results = []
        
        # Traiter le nouveau format avec pages intégrées
        if has_new_format:
            print("🔍 Nouveau format détecté, extraction des sources...")
            
            # Extraire les passages avec pages (patterns plus flexibles)
            page_patterns = [
                r'📄 \*\*Page (\d+)\*\*:\s*(.+?)\s*\[SOURCE-(\d+)\]',  # Format markdown strict
                r'📄 Page (\d+):\s*(.+?)\s*\[SOURCE-(\d+)\]',  # Format sans markdown
                r'Page (\d+):\s*(.+?)\s*\[SOURCE-(\d+)\]'  # Format simple
            ]
            
            page_matches = []
            for pattern in page_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    page_matches.extend(matches)
                    print(f"🎯 Pattern trouvé: {pattern} - {len(matches)} matches")
                    break
            
            # Si pas de matches avec le nouveau format, essayer de parser différemment
            if not page_matches:
                print("⚠️ Aucun match avec les patterns de pages, tentative d'extraction générale...")
                # Essayer d'extraire toutes les citations [SOURCE-X] avec le contexte
                general_pattern = r'(.{50,500}?)\s*\[SOURCE-(\d+)\]'
                general_matches = re.findall(general_pattern, response, re.DOTALL)
                
                for content, source_num in general_matches:
                    # Essayer d'extraire la page du contexte
                    page_in_content = re.search(r'page (\d+)', content, re.IGNORECASE)
                    page = page_in_content.group(1) if page_in_content else "?"
                    
                    page_matches.append((page, content.strip(), source_num))
                    print(f"🔍 Extraction générale: Page {page}, Source {source_num}")
            
            for page, content, source_num in page_matches:
                # Nettoyer le contenu
                clean_content = re.sub(r'\s+', ' ', content.strip())
                
                results.append({
                    "content": clean_content,
                    "metadata": {
                        "source": "Document PDF",  # Sera mis à jour depuis les références
                        "page": str(page),
                        "relevance_score": 85.0,  # Score par défaut
                        "citation_index": int(source_num)
                    }
                })
                print(f"✅ Source extraite: Page {page}, Source {source_num}")
            
            # Extraire les références complètes (patterns plus flexibles)
            ref_patterns = [
                r'\[SOURCE-(\d+)\]\s*(.+?),\s*page\s*(\d+)(?:\s*\(pertinence:\s*(\d+)%\))?',  # Nouveau format
                r'\[SOURCE-(\d+)\]\s*(.+?),\s*page\s*(\d+)',  # Format simple
            ]
            
            ref_matches = []
            for pattern in ref_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    ref_matches.extend(matches)
                    print(f"🎯 Références trouvées: {len(matches)} matches")
                    break
            
            # Associer les références aux résultats
            for match in ref_matches:
                if len(match) >= 3:  # Au moins source_num, source_file, page
                    source_num = int(match[0])
                    source_file = match[1].strip()
                    page = match[2]
                    relevance = match[3] if len(match) > 3 and match[3] else None
                    
                    for result in results:
                        if result["metadata"]["citation_index"] == source_num:
                            result["metadata"]["source"] = source_file
                            if relevance:
                                result["metadata"]["relevance_score"] = float(relevance)
                            print(f"✅ Référence associée: {source_file}, page {page}")
                            break
        
        # Traiter l'ancien format (rétrocompatibilité)
        elif has_old_format:
            parts = response.split("RÉFÉRENCES DOCUMENTAIRES:")
            if len(parts) < 2:
                return None
            
            references_section = parts[1].strip()
            lines = references_section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Pattern pour [SOURCE-X] nom_fichier, page Y
                source_match = re.match(r'\[SOURCE-(\d+)\]\s*(.+?),\s*page\s*(\d+)', line)
                if source_match:
                    source_num = int(source_match.group(1))
                    source_file = source_match.group(2).strip()
                    page = source_match.group(3)
                    
                    # Chercher le contenu correspondant dans le texte principal
                    content = _find_source_content_in_response(response, source_num)
                    
                    results.append({
                        "content": content or f"Contenu de la source {source_num}",
                        "metadata": {
                            "source": source_file,
                            "page": page,
                            "relevance_score": 85.0,  # Score par défaut
                            "citation_index": source_num
                        }
                    })
        
        if results:
            print(f"✅ Extracted {len(results)} sources from text references")
            return {"results": results}
    
    except Exception as e:
        print(f"⚠️ Error extracting sources from text: {e}")
    
    return None


def _find_source_content_in_response(response: str, source_num: int) -> str:
    """Trouve le contenu associé à une source dans la réponse."""
    # Nouveau format avec pages intégrées (priorité)
    page_pattern = rf'📄 \*\*Page \d+\*\*:\s*(.+?)\s*\[SOURCE-{source_num}\]'
    page_match = re.search(page_pattern, response, re.DOTALL)
    if page_match:
        content = page_match.group(1).strip()
        # Nettoyer le contenu
        content = re.sub(r'\n+', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        return content[:500]  # Limiter la longueur
    
    # Anciens formats (rétrocompatibilité)
    patterns = [
        rf'•\s*(.+?)\s*\[SOURCE-{source_num}\]',
        rf'(.+?)\s*\[SOURCE-{source_num}\]',
        rf'(.+?)\s*\|SOURCE-{source_num}\]',  # Format déformé
        rf'(.+?)\s*\|\^{source_num-1}\]'  # Format très déformé
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Prendre le match le plus long (probablement le plus pertinent)
            content = max(matches, key=len).strip()
            # Nettoyer le contenu
            content = re.sub(r'\n+', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            return content[:500]  # Limiter la longueur
    
    return ""


def _clean_response_from_embedded_sources(response: str) -> str:
    """Nettoie la réponse en enlevant le bloc de sources embarquées."""
    if not response:
        return response
    
    # Enlever le bloc de sources embarquées
    import re
    pattern = r'```streamlit_rag_display\n.*?\n```'
    clean_response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # Nettoyer les lignes vides en trop
    clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)
    
    return clean_response.strip()


def _is_rag_query(prompt: str) -> bool:
    """Détermine si une requête est potentiellement une requête RAG."""
    rag_keywords = [
        'pdf', 'document', 'texte', 'page', 'fichier', 'source',
        'selon', 'dans le document', 'cite', 'référence', 'extrait',
        'passage', 'mentionne', 'indique', 'précise', 'explique',
        'classification', 'risque', 'procédure', 'contrôle', 'audit',
        'que dit', 'que précise', 'comment', 'pourquoi', 'qu\'est-ce que'
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in rag_keywords)


def _contains_source_citations(response: str) -> bool:
    """Vérifie si une réponse contient des citations de sources."""
    if not response:
        return False
    
    # Chercher les patterns de citation (y compris les formats déformés et le nouveau format)
    citation_patterns = [
        r'\[SOURCE-\d+\]',  # Format normal
        r'\[\d+\]',         # Citations numérotées
        r'\|SOURCE-\d+\]',  # Format déformé avec pipe
        r'\|\^\d+\]',       # Format très déformé
        r'\|\d+\]',         # Format déformé simple
        r'source\s*:\s*',   # Références textuelles
        r'références\s*documentaires',  # Ancienne section de références
        r'📚 \*\*RÉFÉRENCES COMPLÈTES:\*\*',  # Nouvelle section de références
        r'📄 \*\*Page \d+\*\*:',  # Nouveau format avec pages
        r'⚠️.*citations.*doivent être préservées'
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            print(f"🔍 Citation pattern detected: {pattern}")
            return True
    
    return False


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


def _display_saved_rag_sources():
    """
    Affiche automatiquement les sources RAG sauvegardées s'il y en a.
    """
    try:
        # Importer la fonction depuis rag_tools
        from agents.tools.rag_tools import get_latest_rag_sources
        
        print("🔍 Tentative de récupération des sources sauvegardées...")
        sources_data = get_latest_rag_sources()
        
        if sources_data and sources_data.get("results"):
            print(f"📚 Affichage des sources sauvegardées: {len(sources_data['results'])} sources")
            print(f"📚 Query: {sources_data.get('query', 'N/A')}")
            print(f"📚 Filename: {sources_data.get('filename', 'N/A')}")
            
            # Utiliser notre système d'affichage existant
            query = sources_data.get("query", "")
            
            # Afficher les sources avec le système RAG display
            display_notebooklm_response("", sources_data, query)
            
            return True
        else:
            print("⚠️ Aucune source sauvegardée trouvée ou données vides")
            if sources_data:
                print(f"🔍 Données récupérées: {sources_data}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage des sources sauvegardées: {e}")
        import traceback
        traceback.print_exc()
        return False


def _is_rag_response(response: str) -> bool:
    """
    Détermine si une réponse provient d'une requête RAG (mention de pages).
    """
    if not response:
        return False
    
    # Chercher des mentions de pages qui indiquent une réponse RAG
    page_indicators = [
        r'page \d+',
        r'à la page',
        r'selon le document',
        r'dans le document',
        r'comme mentionné',
        r'détaillé',
        r'spécifié'
    ]
    
    response_lower = response.lower()
    for indicator in page_indicators:
        if re.search(indicator, response_lower):
            return True
    
    return False 


# Fonction de debug supprimée - plus nécessaire 