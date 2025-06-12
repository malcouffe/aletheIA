"""
Affichage style NotebookLM pour les réponses RAG.
Interface unifiée qui lie la réponse de l'agent aux sources avec des citations numérotées.
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Optional


def display_rag_results(results_json: str, query: str = "") -> None:
    """
    Affichage style NotebookLM des résultats de recherche PDF.
    
    Args:
        results_json: JSON string contenant les résultats structurés
        query: Requête originale de l'utilisateur
    """
    try:
        results = json.loads(results_json)
    except json.JSONDecodeError:
        st.error("❌ Erreur de parsing des résultats")
        return
    
    if not results.get("success"):
        _display_error_simple(results)
        return
    
    _display_notebooklm_style(results, query)


def display_notebooklm_response(agent_response: str, sources_data: Dict, query: str = "") -> None:
    """
    Affichage principal style NotebookLM avec réponse intégrée et sources numérotées.
    
    Args:
        agent_response: Réponse de l'agent avec citations [1], [2], etc.
        sources_data: Données des sources parsed du JSON
        query: Requête originale
    """
    st.markdown("---")
    
    # 1. Réponse de l'agent avec citations intégrées
    st.markdown("### 💬 Réponse")
    
    # Parse les citations dans la réponse et les rend cliquables
    enhanced_response = _enhance_citations_in_response(agent_response, sources_data)
    st.markdown(enhanced_response, unsafe_allow_html=True)
    
    # 2. Sources numérotées correspondantes
    st.markdown("### 📚 Sources")
    _display_numbered_sources(sources_data, query)
    
    st.markdown("---")


def _enhance_citations_in_response(response: str, sources_data: Dict) -> str:
    """
    Améliore les citations [1], [2] dans la réponse pour les rendre visuelles.
    """
    if not response:
        return "Aucune réponse disponible."
    
    # Trouve toutes les citations [1], [2], etc.
    citation_pattern = r'\[(\d+)\]'
    citations = re.findall(citation_pattern, response)
    
    # Remplace chaque citation par une version stylée
    enhanced_response = response
    for citation_num in set(citations):
        old_citation = f"[{citation_num}]"
        new_citation = f'<span style="background-color: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">[{citation_num}]</span>'
        enhanced_response = enhanced_response.replace(old_citation, new_citation)
    
    return enhanced_response


def _display_numbered_sources(sources_data: Dict, query: str) -> None:
    """
    Affiche les sources numérotées de façon claire et concise.
    """
    search_results = sources_data.get("results", [])
    
    if not search_results:
        st.info("📖 Aucune source disponible")
        return
    
    # Affiche chaque source avec son numéro
    for i, result in enumerate(search_results[:5], 1):  # Limite à 5 sources
        _display_single_numbered_source(result, i)


def _display_single_numbered_source(result: Dict[str, Any], index: int) -> None:
    """
    Affiche une source numérotée de façon concise style NotebookLM.
    """
    metadata = result.get("metadata", {})
    content = result.get("content", "")
    
    # Informations essentielles
    source = metadata.get("source", "Document inconnu")
    page = metadata.get("page", "?")
    
    # Nettoyage du nom de fichier
    if source.endswith('.pdf'):
        source = source[:-4]
    
    # Extrait concis (1-2 phrases max)
    excerpt = _extract_key_sentences(content, max_sentences=1)
    
    # Style NotebookLM : numéro + source + extrait
    with st.container():
        # Citation numérotée avec style
        col1, col2 = st.columns([1, 15])
        
        with col1:
            st.markdown(f"**[{index}]**")
        
        with col2:
            # Nom du document et page
            st.markdown(f"**{source}**, page {page}")
            
            # Extrait avec style citation
            st.markdown(f'*"{excerpt}"*')
            
            # Bouton discret pour voir plus
            if len(content) > len(excerpt) + 50:
                with st.expander("👁️ Voir le passage complet", expanded=False):
                    st.text_area(
                        "Contexte complet",
                        value=content,
                        height=200,
                        key=f"source_content_{index}",
                        label_visibility="collapsed"
                    )
        
        st.markdown("")  # Espacement


def _display_notebooklm_style(results: Dict[str, Any], query: str) -> None:
    """
    Affichage style NotebookLM pour les résultats existants (fallback).
    """
    search_results = results.get("results", [])
    
    if not search_results:
        st.info("📖 Aucune source trouvée")
        return
    
    # En-tête simple
    st.markdown("### 📚 Sources trouvées")
    if query:
        st.markdown(f"*Pour la recherche : {query}*")
    st.markdown("")
    
    # Affichage des sources numérotées
    for i, result in enumerate(search_results[:5], 1):
        _display_single_numbered_source(result, i)


def _display_error_simple(results: Dict[str, Any]) -> None:
    """Affichage simplifié des erreurs."""
    error_message = results.get("error", "Erreur inconnue")
    st.error(f"**Recherche échouée:** {error_message}")
    
    if "no_context" in results.get("error_type", ""):
        st.info("💡 **Astuce:** Chargez d'abord un fichier PDF via la barre latérale")


def _extract_key_sentences(text: str, max_sentences: int = 1) -> str:
    """
    Extrait les phrases les plus importantes du texte.
    Optimisé pour les citations courtes style NotebookLM.
    """
    if not text:
        return "Contenu non disponible"
    
    # Divise en phrases
    sentences = []
    for sep in ['. ', '! ', '? ']:
        if sep in text:
            sentences = text.split(sep)
            break
    
    if not sentences:
        # Si pas de séparateurs, prend les premiers 150 caractères
        return text[:150] + "..." if len(text) > 150 else text
    
    # Prend les premières phrases significatives
    key_sentences = []
    total_length = 0
    
    for sentence in sentences[:max_sentences]:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Évite les phrases trop courtes
            key_sentences.append(sentence)
            total_length += len(sentence)
            
            # Limite la longueur totale pour rester lisible
            if total_length > 200:
                break
    
    if not key_sentences:
        # Fallback: premiers mots
        words = text.split()[:20]
        return ' '.join(words) + "..." if len(words) == 20 else ' '.join(words)
    
    result = '. '.join(key_sentences)
    
    # Ajoute un point final si nécessaire
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result


def display_structured_rag_response(response: str, query: str = "") -> bool:
    """
    Détecte et affiche les réponses RAG structurées style NotebookLM.
    
    Args:
        response: Réponse de l'agent (JSON ou texte)
        query: Requête originale
        
    Returns:
        True si affiché comme résultat structuré, False sinon
    """
    # Détecte les citations dans la réponse (style NotebookLM)
    citation_pattern = r'\[(\d+)\]'
    has_citations = bool(re.search(citation_pattern, response))
    
    # Détecte le JSON dans la réponse
    try:
        # Cherche du JSON dans un bloc de code
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_part = response[start:end].strip()
            
            # Parse les sources
            sources_data = json.loads(json_part)
            
            if isinstance(sources_data, dict) and sources_data.get("success") and sources_data.get("results"):
                if has_citations:
                    # Style NotebookLM complet avec citations intégrées
                    # Extrait la partie texte de la réponse (avant le JSON)
                    text_part = response[:response.find("```json")].strip()
                    display_notebooklm_response(text_part, sources_data, query)
                else:
                    # Fallback: affichage standard des sources
                    display_rag_results(json_part, query)
                return True
                
        elif response.strip().startswith("{") and response.strip().endswith("}"):
            # Toute la réponse est en JSON
            sources_data = json.loads(response.strip())
            if isinstance(sources_data, dict) and ("success" in sources_data or "results" in sources_data):
                display_rag_results(response.strip(), query)
                return True
            
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    return False 