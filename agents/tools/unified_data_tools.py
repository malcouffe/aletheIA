"""
Unified Data Tools for Data Analysis Agents
Combines data loading, discovery, and visualization capabilities following smolagents best practices.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from smolagents import tool
from ..config.agent_config import VISUALIZATION_CONFIG


@tool
def load_and_explore_csv(filename: str, explore: bool = True) -> str:
    """
    Trouve un fichier CSV et indique à l'agent comment le charger.
    
    Args:
        filename: Nom du fichier CSV à trouver (ex: "titanic.csv")
        explore: Si True, suggère aussi des commandes d'exploration (défaut: True)
    
    Returns:
        Instructions simples pour charger le fichier CSV trouvé
    """
    try:
        # Recherche du fichier dans data/csv_temp
        search_dirs = ['data/csv_temp', './data/csv_temp']
        found_path = None
        
        for directory in search_dirs:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    if filename in files:
                        found_path = os.path.join(root, filename)
                        break
                if found_path:
                    break
        
        if not found_path:
            # Lister les fichiers disponibles
            available_files = []
            for directory in search_dirs:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        csv_files = [f for f in files if f.endswith('.csv')]
                        available_files.extend(csv_files)
            
            if available_files:
                files_list = ', '.join(available_files)
                return f"❌ Fichier '{filename}' non trouvé. Fichiers CSV disponibles: {files_list}"
            else:
                return "❌ Aucun fichier CSV trouvé dans data/csv_temp/"
        
        # Créer un nom de variable simple
        var_name = filename.replace('.csv', '').replace('-', '_').replace(' ', '_') + '_df'
        
        # Instructions simples pour l'agent
        instructions = f"✅ Fichier trouvé ! Pour charger '{filename}', utilise cette commande:\n\n"
        instructions += f"{var_name} = pd.read_csv(r'{found_path}')"
        
        if explore:
            instructions += f"\n\nPour explorer les données:\n"
            instructions += f"• {var_name}.head() - Voir les premières lignes\n"
            instructions += f"• {var_name}.info() - Infos sur les colonnes\n"
            instructions += f"• {var_name}.describe() - Statistiques descriptives\n"
            instructions += f"• {var_name}.shape - Taille du dataset\n\n"
            instructions += f"💡 Conseil: Utilise display_figures() pour afficher tes graphiques et réponds toujours en langage naturel !"
        
        return instructions
        
    except Exception as e:
        return f"❌ Erreur lors de la recherche de '{filename}': {str(e)}"


@tool
def display_figures(figures_dict: Dict[str, Any], figure_type: str = "auto") -> str:
    """
    Outil d'affichage unifié pour les graphiques matplotlib et plotly.
    
    BONNES PRATIQUES SMOLAGENTS INTÉGRÉES:
    - Gestion d'erreurs détaillée avec messages clairs
    - Nettoyage automatique de la mémoire
    - Validation robuste des entrées
    
    Args:
        figures_dict: Dictionnaire associant les noms de graphiques aux objets graphiques.
                     Maximum 10 graphiques par appel pour éviter les problèmes de mémoire.
                     Utilise des noms descriptifs: {"analyse_ventes": fig1, "correlation": fig2}
                     
        figure_type: Type de graphiques à afficher:
                    - "matplotlib": Pour les graphiques matplotlib/seaborn
                    - "plotly": Pour les graphiques plotly
                    - "auto": (défaut) Détecte automatiquement le type de graphique
    
    Returns:
        Message de statut détaillé avec informations de débogage selon les bonnes pratiques smolagents.
    """
    try:
        # Validation approfondie avec messages informatifs (bonne pratique smolagents)
        if not isinstance(figures_dict, dict):
            error_msg = f"❌ ERREUR TYPE: Attendu dict, reçu {type(figures_dict)}"
            return error_msg + "\\n\\n🔧 Utilise un dictionnaire: display_figures({'nom': figure_object})"
        
        if not figures_dict:
            warning_msg = "⚠️ ATTENTION: Dictionnaire de graphiques vide"
            return warning_msg + "\\n\\n🔧 Crée d'abord tes graphiques avec matplotlib ou plotly"
        
        # Vérifier la limite de graphiques
        max_figures = VISUALIZATION_CONFIG.get("max_figures_per_call", 10)
        if len(figures_dict) > max_figures:
            error_msg = f"❌ LIMITE DÉPASSÉE: {len(figures_dict)} graphiques (max {max_figures})"
            return error_msg + f"\\n\\n🔧 Divise en plusieurs appels avec ≤{max_figures} graphiques"
        
        displayed_count = 0
        failed_figures = []
        
        for fig_name, fig_obj in figures_dict.items():
            try:
                # Déterminer le type de graphique si auto
                current_figure_type = figure_type
                if current_figure_type == "auto":
                    if isinstance(fig_obj, go.Figure):
                        current_figure_type = "plotly"
                    else:
                        current_figure_type = "matplotlib"
                
                # Gérer les graphiques matplotlib
                if current_figure_type == "matplotlib":
                    # Gérer les objets axes en récupérant leur figure
                    if hasattr(fig_obj, 'get_figure'):
                        fig_obj = fig_obj.get_figure()
                    
                    if hasattr(fig_obj, 'savefig'):
                        st.pyplot(fig_obj)
                        plt.close(fig_obj)  # Libérer la mémoire (bonne pratique)
                        displayed_count += 1
                    else:
                        failed_figures.append(f"{fig_name}: Objet matplotlib invalide")
                
                # Gérer les graphiques plotly
                elif current_figure_type == "plotly":
                    if isinstance(fig_obj, go.Figure):
                        st.plotly_chart(fig_obj, use_container_width=True)
                        displayed_count += 1
                    else:
                        failed_figures.append(f"{fig_name}: Objet plotly invalide")
                
                else:
                    failed_figures.append(f"{fig_name}: Type invalide '{current_figure_type}'")
                    
            except Exception as e:
                failed_figures.append(f"{fig_name}: {str(e)}")
                st.error(f"Erreur graphique {fig_name}: {str(e)}")
        
        # Message de résultat détaillé (bonne pratique smolagents)
        result_parts = [f"📊 RÉSULTATS D'AFFICHAGE:"]
        result_parts.append(f"✅ {displayed_count} graphique(s) affiché(s) avec succès")
        
        if failed_figures:
            result_parts.append(f"❌ {len(failed_figures)} graphique(s) ont échoué:")
            for failure in failed_figures:
                result_parts.append(f"   • {failure}")
            result_parts.append("\\n🔧 CONSEILS DE DÉBOGAGE:")
            result_parts.append("   • Vérifies-tu que tes objets graphiques sont valides ?")
            result_parts.append("   • Pour matplotlib: utilise fig, ax = plt.subplots()")
            result_parts.append("   • Pour plotly: utilise fig = go.Figure()")
        
        if displayed_count > 0:
            result_parts.append(f"\\n🧹 Mémoire libérée pour {displayed_count} graphique(s)")
        
        final_result = "\\n".join(result_parts)
        return final_result
        
    except Exception as e:
        error_msg = f"💥 ERREUR CRITIQUE: {str(e)}"
        return error_msg + "\\n\\n🔧 Assure-toi de passer des objets graphiques valides dans un dictionnaire"


# ============ OUTILS DÉPRÉCIÉS - COMPATIBILITÉ ASCENDANTE ============
# Ces outils restent pour la compatibilité mais ne devraient plus être utilisés

@tool  
def data_loader(file_context: str, mode: str = "auto") -> str:
    """
    [DÉPRÉCIÉ] Outil legacy pour compatibilité ascendante.
    
    ⚠️ UTILISE PLUTÔT: load_and_explore_csv() qui suit les bonnes pratiques smolagents.
    
    Cet outil reste disponible uniquement pour compatibilité avec l'ancien code.
    Il redirige vers le nouvel outil unifié load_and_explore_csv.
    
    Args:
        file_context: Nom du fichier CSV à charger ou contexte de recherche
        mode: Mode de fonctionnement (défaut: "auto")
    
    Returns:
        Code Python généré pour charger ou découvrir les fichiers CSV
    """
    # Rediriger vers le nouvel outil unifié
    if file_context.endswith('.csv'):
        return load_and_explore_csv(file_context, explore=True)
    else:
        # Mode découverte - générer du code de découverte
        return '''# 🔍 DÉCOUVERTE DES FICHIERS CSV DISPONIBLES
import os

print("📁 Recherche des fichiers CSV...")
found_files = []
search_dirs = ['data/csv_temp', './data/csv_temp']

for directory in search_dirs:
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            csv_files = [f for f in files if f.endswith('.csv')]
            for csv_file in csv_files:
                found_files.append((csv_file, os.path.join(root, csv_file)))

if found_files:
    print("✅ Fichiers CSV disponibles:")
    for i, (filename, filepath) in enumerate(found_files, 1):
        size = os.path.getsize(filepath) / 1024
        print(f"  {i}. {filename} ({size:.1f} KB)")
    print("\\n🚀 Pour charger un fichier:")
    print("   • Utilise load_and_explore_csv('nom_fichier.csv')")
else:
    print("❌ Aucun fichier CSV trouvé dans data/csv_temp/")'''


@tool
def get_dataframe(dataset_name: str) -> str:
    """
    [DÉPRÉCIÉ] Outil legacy pour compatibilité ascendante.
    
    ⚠️ UTILISE PLUTÔT: load_and_explore_csv() qui gère tout automatiquement.
    
    Cet outil reste disponible uniquement pour compatibilité avec l'ancien code.
    
    Args:
        dataset_name: Nom du dataset à récupérer depuis la session Streamlit
    
    Returns:
        Code Python généré pour accéder au dataset ou message d'erreur
    """
    # Code de compatibilité minimal
    return f'''# ⚠️ FONCTION DÉPRÉCIÉE - get_dataframe()
import streamlit as st

# Vérifier si le dataset existe en session
df_key = 'dataframe_{dataset_name}'

if 'st' in globals() and hasattr(st, 'session_state') and df_key in st.session_state:
    {dataset_name}_df = st.session_state[df_key]
    print(f"✅ Dataset '{dataset_name}' récupéré depuis la session")
    print(f"📏 Forme: {{{dataset_name}_df.shape[0]}} lignes × {{{dataset_name}_df.shape[1]}} colonnes")
    print(f"🎯 Variable disponible: {dataset_name}_df")
else:
    print(f"❌ Dataset '{dataset_name}' non trouvé en session")
    print("🚀 SOLUTION: Utilise load_and_explore_csv('fichier.csv') à la place")
    print("   Cette méthode est plus fiable et suit les bonnes pratiques")'''


@tool
def load_csv_data(filename: str) -> str:
    """
    [DÉPRÉCIÉ] Outil legacy pour compatibilité ascendante.
    
    ⚠️ UTILISE PLUTÔT: load_and_explore_csv() qui est plus complet et optimisé.
    
    Cet outil reste disponible uniquement pour compatibilité avec l'ancien code.
    Il redirige vers le nouvel outil unifié.
    
    Args:
        filename: Nom du fichier CSV à charger
    
    Returns:
        Code Python généré pour charger le fichier CSV
    """
    # Rediriger vers le nouvel outil unifié (sans exploration par défaut pour compatibilité)
    return load_and_explore_csv(filename, explore=False) 