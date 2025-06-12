"""
Affichage style NotebookLM pour les réponses RAG.
Interface unifiée qui lie la réponse de l'agent aux sources avec des citations numérotées.
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Optional


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
        st.info("📚 Aucune source trouvée pour cette recherche.")
        return
    
    # Grouper par fichier pour un affichage organisé
    sources_by_file = {}
    for i, result in enumerate(search_results, 1):
        # Accéder correctement aux métadonnées
        metadata = result.get("metadata", {})
        filename = metadata.get("source", "Document inconnu")
        
        if filename not in sources_by_file:
            sources_by_file[filename] = []
        sources_by_file[filename].append((i, result))
    
    # Affichage compact et élégant
    with st.expander(f"🔍 **{len(search_results)} source(s) trouvée(s)**", expanded=False):
        for filename, file_sources in sources_by_file.items():
            st.markdown(f"**📄 {filename.replace('.pdf', '')}**")
            
            # Affichage en colonnes pour un look plus compact
            for citation_num, result in file_sources:
                metadata = result.get("metadata", {})
                page = metadata.get("page", "?")
                content_preview = result.get("content", "")[:150]  # Plus court
                relevance_score = metadata.get("relevance_score", 0)
                
                # Style carte compact
                st.markdown(f"""
                <div style="
                    background-color: #f8f9fa; 
                    border-left: 4px solid #1976d2; 
                    padding: 10px; 
                    margin: 8px 0; 
                    border-radius: 4px;
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <span style="
                            background-color: #1976d2; 
                            color: white; 
                            padding: 2px 8px; 
                            border-radius: 12px; 
                            font-size: 0.8em; 
                            font-weight: bold; 
                            margin-right: 10px;
                        ">[{citation_num}]</span>
                        <span style="
                            color: #666; 
                            font-size: 0.9em;
                            font-weight: bold;
                        ">Page {page}</span>
                        <span style="
                            color: #888; 
                            font-size: 0.8em; 
                            margin-left: 10px;
                        ">Pertinence: {relevance_score:.1f}%</span>
                    </div>
                    <div style="
                        color: #333; 
                        font-size: 0.9em; 
                        font-style: italic; 
                        line-height: 1.4;
                    ">"{content_preview}..."</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Espacement entre fichiers


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
    citations_pattern = r'\[\d+\]'
    has_citations = bool(re.search(citations_pattern, response))
    
    # Essaye d'extraire le JSON (différents formats possibles)
    json_data = None
    
    # Format 1: Bloc JSON markdown
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Format 2: JSON brut à la fin
    if not json_data:
        lines = response.strip().split('\n')
        for i in range(len(lines)-1, -1, -1):
            line = lines[i].strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    json_data = json.loads('\n'.join(lines[i:]))
                    break
                except json.JSONDecodeError:
                    continue
    
    # Validation : vérifier que c'est bien du RAG structuré
    is_valid_rag = (
        json_data and 
        isinstance(json_data, dict) and
        json_data.get("success") and
        json_data.get("results") and
        len(json_data.get("results", [])) > 0
    )
    
    if has_citations and is_valid_rag:
        # Séparer la réponse du JSON pour l'affichage
        if json_match:
            clean_response = response[:json_match.start()].strip()
        else:
            # Pour le JSON brut, couper avant la ligne JSON
            lines = response.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    clean_response = '\n'.join(lines[:i]).strip()
                    break
            else:
                clean_response = response
        
        # Afficher via display_notebooklm_response
        display_notebooklm_response(clean_response, json_data, query)
        return True
    
    return False

 