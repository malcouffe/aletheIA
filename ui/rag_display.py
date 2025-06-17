"""
Affichage RAG SIMPLIFIÉ - Interface claire et robuste.
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Optional
import time


def display_notebooklm_response(agent_response: str, sources_data: Dict, query: str = "") -> None:
    """
    Affichage principal des sources RAG - VERSION SIMPLIFIÉE.
    """
    print(f"🎨 RAG Display: Starting simplified display...")
    
    # Validation simple
    if not sources_data or not isinstance(sources_data, dict):
        st.error("❌ Données sources invalides")
        st.markdown(agent_response)
        return
    
    results = sources_data.get("results", [])
    if not results:
        st.info("📚 Aucune source trouvée")
        st.markdown(agent_response)
        return
    
    print(f"✅ RAG Display: Displaying {len(results)} sources")
    
    st.markdown("---")
    
    # Afficher la réponse seulement si elle contient du contenu utile
    if agent_response and agent_response.strip() and agent_response.strip() != "Aucune réponse disponible." and len(agent_response.strip()) > 10:
        st.markdown("### 💬 Réponse")
        enhanced_response = _highlight_citations(agent_response)
        st.markdown(enhanced_response, unsafe_allow_html=True)
        st.markdown("---")
    
    # Sources dans un expander
    st.markdown("### 📚 Sources")
    
    # Bouton pour copier toutes les citations
    if st.button("📋 Copier toutes les citations", key="copy_all_citations"):
        all_citations = _generate_citations_text(results)
        st.code(all_citations, language="text")
        st.success("✅ Toutes les citations affichées ci-dessus - vous pouvez les copier !")
    
    with st.expander(f"🔍 **{len(results)} source(s) trouvée(s)** - Cliquez pour voir les passages complets", expanded=True):
        _display_simple_sources(results)
    
    st.markdown("---")
    print(f"✅ RAG Display: Display completed")


# Fonction _display_citations_summary supprimée - cause du HTML bugué


def _generate_citations_text(results: List[Dict]) -> str:
    """Génère un texte formaté des citations pour copier-coller."""
    if not results:
        return "Aucune citation disponible."
    
    citations_text = "CITATIONS ET SOURCES :\n\n"
    
    for i, result in enumerate(results, 1):
        try:
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            source = metadata.get("source", f"Document {i}")
            
            citations_text += f"[{i}] {source}, page {page}\n"
            citations_text += f'    "{content}"\n\n'
            
        except Exception as e:
            citations_text += f"[{i}] Erreur: {e}\n\n"
    
    return citations_text


def _highlight_citations(response: str) -> str:
    """Colore les citations [1], [2], [SOURCE-1], [SOURCE-2], etc. et corrige les citations déformées."""
    if not response:
        return "Aucune réponse disponible."
    
    # D'abord, nettoyer et corriger les citations déformées
    result = response
    
    # Corriger les citations déformées courantes
    # |SOURCE-1] → [SOURCE-1]
    result = re.sub(r'\|SOURCE-(\d{1,2})\]', r'[SOURCE-\1]', result)
    # |^0] → [1], |^1] → [2], etc. (caractères corrompus)
    result = re.sub(r'\|\^(\d{1,2})\]', lambda m: f'[{int(m.group(1)) + 1}]', result)
    # |1] → [1], |2] → [2] (pipe au lieu de crochet ouvrant)
    result = re.sub(r'\|(\d{1,2})\]', r'[\1]', result)
    
    # Améliorer l'affichage du nouveau format avec pages
    # Convertir les pages markdown en HTML stylé
    result = re.sub(r'📄 \*\*Page (\d+)\*\*:', 
                   r'<div style="background: #f0f8ff; padding: 8px; border-left: 3px solid #1976d2; margin: 10px 0; border-radius: 5px;"><strong style="color: #1976d2;">📄 Page \1:</strong></div>', 
                   result)
    
    # Améliorer l'affichage de la section références
    result = re.sub(r'📚 \*\*RÉFÉRENCES COMPLÈTES:\*\*', 
                   r'<div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 15px 0;"><strong style="color: #2e7d32;">📚 RÉFÉRENCES COMPLÈTES:</strong></div>', 
                   result)
    
    # Regex pour [chiffre] et [SOURCE-chiffre]
    citation_patterns = [
        (r'\[(\d{1,2})\]', lambda m: f'<span style="background-color: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 8px; font-weight: bold; margin: 0 2px;">[{m.group(1)}]</span>'),
        (r'\[SOURCE-(\d{1,2})\]', lambda m: f'<span style="background-color: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 8px; font-weight: bold; margin: 0 2px;">[SOURCE-{m.group(1)}]</span>')
    ]
    
    for pattern, replacement in citation_patterns:
        result = re.sub(pattern, replacement, result)
    
    return result


def _display_simple_sources(results: List[Dict]) -> None:
    """Affiche les sources de manière simple et claire avec passages complets."""
    for i, result in enumerate(results, 1):
        try:
            # Extraction sécurisée des données
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            source = metadata.get("source", f"Document {i}")
            relevance = metadata.get("relevance_score", 0)
            
            # Échapper les caractères HTML dans le contenu
            import html
            safe_content = html.escape(content)
            safe_source = html.escape(str(source))
            
            # Affichage amélioré avec meilleure mise en valeur des passages
            source_html = f"""
            <div style="background: #f8f9fa; border: 2px solid #1976d2; padding: 20px; margin: 15px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="margin-bottom: 15px; display: flex; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <span style="background: #1976d2; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 1.1em;">SOURCE [{i}]</span>
                    <strong style="color: #333; font-size: 1.1em;">📄 {safe_source}</strong>
                    <span style="color: #666; font-size: 0.95em;">📖 Page {page}</span>
                    <span style="color: #28a745; font-weight: bold; font-size: 0.9em;">🎯 Pertinence: {relevance:.0f}%</span>
                </div>
                <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e0e0e0; border-left: 4px solid #1976d2;">
                    <div style="margin-bottom: 10px;">
                        <strong style="color: #1976d2; font-size: 1.1em;">📖 PASSAGE CORRESPONDANT :</strong>
                    </div>
                    <div style="background: #f8f9ff; padding: 15px; border-radius: 5px; font-family: 'Georgia', serif; line-height: 1.6; font-size: 1.05em; color: #2c3e50; border-left: 3px solid #3498db;">
                        <em>"{safe_content}"</em>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(source_html, unsafe_allow_html=True)
            
            # Ajouter un bouton pour copier le passage
            if st.button(f"📋 Copier le passage {i}", key=f"copy_passage_{i}"):
                st.code(content, language="text")
                st.success(f"✅ Passage {i} affiché ci-dessus - vous pouvez le copier !")
            
        except Exception as e:
            st.error(f"❌ Erreur source [{i}]: {e}")


def display_enhanced_rag_response(agent_response: str, sources_data: Dict, query: str = "") -> None:
    """
    Affichage RAG AMÉLIORÉ avec passages mis en évidence et meilleure organisation.
    """
    print(f"🎨 RAG Display: Starting ENHANCED display...")
    
    # Validation simple
    if not sources_data or not isinstance(sources_data, dict):
        st.error("❌ Données sources invalides")
        st.markdown(agent_response)
        return
    
    results = sources_data.get("results", [])
    if not results:
        st.info("📚 Aucune source trouvée")
        st.markdown(agent_response)
        return
    
    print(f"✅ RAG Display: Displaying {len(results)} sources with enhanced format")
    
    # En-tête avec métadonnées
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**🔍 Requête analysée :** {query}")
    with col2:
        st.markdown(f"**📚 Sources trouvées :** {len(results)}")
    with col3:
        avg_relevance = sum(r.get("metadata", {}).get("relevance_score", 0) for r in results) / len(results)
        st.markdown(f"**🎯 Pertinence moy. :** {avg_relevance:.0f}%")
    
    # 1. Réponse avec citations colorées
    st.markdown("### 💬 Réponse de l'IA")
    enhanced_response = _highlight_citations(agent_response)
    st.markdown(enhanced_response, unsafe_allow_html=True)
    
    # 2. Affichage des sources avec passages
    st.markdown("### 📚 Sources et Passages du Document")
    
    # Résumé compact des sources
    _display_enhanced_citations_summary(results)
    
    # Sources détaillées avec passages
    with st.expander(f"📖 **PASSAGES COMPLETS des {len(results)} sources** - Cliquez pour voir tous les détails", expanded=True):
        _display_simple_sources(results)
    
    # Section de téléchargement
    st.markdown("### 📥 Exporter les résultats")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Copier toutes les citations", key="copy_all_citations"):
            all_citations = _generate_detailed_citations_text(results, query, agent_response)
            st.code(all_citations, language="text")
            st.success("✅ Toutes les citations formatées sont affichées ci-dessus !")
    
    with col2:
        if st.button("📊 Voir les statistiques des sources", key="show_stats"):
            _display_sources_statistics(results)
    
    st.markdown("---")
    print(f"✅ RAG Display: Enhanced display completed")


def _generate_detailed_citations_text(results: List[Dict], query: str, response: str) -> str:
    """Génère un rapport détaillé avec citations, passages et métadonnées."""
    if not results:
        return "Aucune citation disponible."
    
    detailed_text = f"""RAPPORT DE RECHERCHE DOCUMENTAIRE
=====================================

REQUÊTE : {query}
DATE : {time.strftime('%Y-%m-%d %H:%M:%S')}
NOMBRE DE SOURCES : {len(results)}

RÉPONSE DE L'IA :
{response}

SOURCES ET PASSAGES DÉTAILLÉS :
==============================

"""
    
    for i, result in enumerate(results, 1):
        try:
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            source = metadata.get("source", f"Document {i}")
            relevance = metadata.get("relevance_score", 0)
            
            detailed_text += f"""
SOURCE [{i}] - {source}
{'='*50}
📄 Document : {source}
📖 Page : {page}
🎯 Pertinence : {relevance:.1f}%

📖 PASSAGE :
"{content}"

{'='*50}

"""
            
        except Exception as e:
            detailed_text += f"\nERREUR SOURCE [{i}]: {e}\n"
    
    return detailed_text


def _display_enhanced_citations_summary(results: List[Dict]) -> None:
    """Affiche un résumé amélioré des citations avec plus de détails."""
    if not results:
        return
    
    st.markdown("#### 🔗 Résumé des Citations")
    
    # Créer un résumé plus détaillé
    citations_html = '<div style="background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%); padding: 20px; border-radius: 12px; margin: 15px 0; border: 1px solid #1976d2;">'
    citations_html += '<div style="margin-bottom: 15px;"><strong style="color: #1976d2; font-size: 1.2em;">📋 Références trouvées dans les documents :</strong></div>'
    
    for i, result in enumerate(results, 1):
        try:
            import html
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            source = metadata.get("source", f"Document {i}")
            relevance = metadata.get("relevance_score", 0)
            content_preview = result.get("content", "")[:150] + "..." if len(result.get("content", "")) > 150 else result.get("content", "")
            
            # Échapper les caractères HTML
            safe_source = html.escape(str(source))
            safe_content_preview = html.escape(str(content_preview))
            
            # Couleur basée sur la pertinence
            if relevance >= 90:
                color = "#2e7d32"  # Vert foncé
            elif relevance >= 75:
                color = "#1976d2"  # Bleu
            elif relevance >= 60:
                color = "#f57c00"  # Orange
            else:
                color = "#d32f2f"  # Rouge
            
            citations_html += f'''
            <div style="margin: 12px 0; padding: 15px; background: white; border-radius: 8px; border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
                    <span style="background: {color}; color: white; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.9em;">SOURCE [{i}]</span>
                    <strong style="color: #333;">{safe_source}</strong>
                    <span style="color: #666; font-size: 0.9em;">📖 Page {page}</span>
                    <span style="color: {color}; font-weight: bold; font-size: 0.9em;">🎯 {relevance:.0f}%</span>
                </div>
                <div style="font-size: 0.9em; color: #555; font-style: italic; line-height: 1.4;">
                    <strong>Extrait :</strong> "{safe_content_preview}"
                </div>
            </div>
            '''
        except Exception as e:
            citations_html += f'<div style="color: red; padding: 10px;">Erreur citation [{i}]: {html.escape(str(e))}</div>'
    
    citations_html += '</div>'
    st.markdown(citations_html, unsafe_allow_html=True)


def _display_sources_statistics(results: List[Dict]) -> None:
    """Affiche des statistiques détaillées sur les sources trouvées."""
    if not results:
        st.warning("Aucune source disponible pour les statistiques.")
        return
    
    st.markdown("#### 📊 Statistiques des Sources")
    
    # Calculs statistiques
    relevance_scores = [r.get("metadata", {}).get("relevance_score", 0) for r in results]
    content_lengths = [len(r.get("content", "")) for r in results]
    
    # Affichage en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Nombre de sources", len(results))
        st.metric("🎯 Pertinence moyenne", f"{sum(relevance_scores)/len(relevance_scores):.1f}%")
    
    with col2:
        st.metric("🔝 Pertinence maximale", f"{max(relevance_scores):.1f}%")
        st.metric("📝 Longueur moyenne", f"{sum(content_lengths)//len(content_lengths)} caractères")
    
    with col3:
        st.metric("📄 Pages couvertes", len(set(r.get("metadata", {}).get("page", "?") for r in results)))
        st.metric("📈 Sources > 80%", len([s for s in relevance_scores if s >= 80]))
    
    # Graphique de répartition des scores de pertinence
    if len(relevance_scores) > 1:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame({
            'Source': [f"Source {i+1}" for i in range(len(relevance_scores))],
            'Pertinence': relevance_scores,
            'Longueur': content_lengths
        })
        
        fig = px.bar(df, x='Source', y='Pertinence', 
                     title="Scores de Pertinence par Source",
                     color='Pertinence',
                     color_continuous_scale='Viridis')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True) 