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
def data_loader(file_context: str, mode: str = "auto") -> str:
    """
    Outil de chargement et découverte de données unifié.
    
    Combine les capacités de découverte et de chargement de fichiers avec une sélection 
    intelligente du mode basée sur le contexte d'entrée.
    
    Args:
        file_context: Nom de fichier (ex: "data.csv") ou texte descriptif des données recherchées.
                     Recherche dans le dossier data/csv_temp/ et ses sous-dossiers où sont stockés les fichiers CSV.
                     Utilisez des noms de fichiers exacts pour un chargement plus rapide: "ventes_2024.csv"
        mode: Mode d'opération - "load" (charger un fichier spécifique), "discover" (lister tous les fichiers),
              ou "auto" (par défaut, sélectionne intelligemment le mode selon l'entrée)
    
    Returns:
        Pour mode="load" ou auto-détection load:
            Résumé détaillé des données avec forme, colonnes, aperçu et instructions de chargement.
            Inclut l'analyse des valeurs manquantes et des conseils d'utilisation.
        
        Pour mode="discover" ou auto-détection discovery:
            Liste des fichiers CSV disponibles avec leurs emplacements et informations de base.
    """
    try:
        print(f"Recherche de données avec contexte='{file_context}', mode='{mode}'")
        
        # Déterminer le mode si auto
        if mode == "auto":
            mode = "load" if file_context.endswith('.csv') else "discover"
        
        print(f"Mode sélectionné: {mode}")
        
        # Se concentrer sur le répertoire CSV temp où sont stockés les fichiers utilisateur
        search_dirs = ['data/csv_temp', './data/csv_temp']
        
        if mode == "discover":
            return _discover_files(search_dirs)
        else:  # mode == "load"
            return _load_file(file_context, search_dirs)
            
    except Exception as e:
        error_msg = f"Une erreur s'est produite lors du chargement des données: {str(e)}"
        print(error_msg)
        return error_msg + "\n\nConseil: Essaie data_loader('nom_fichier.csv', mode='discover') pour voir les fichiers disponibles"


def _discover_files(search_dirs: list) -> str:
    """Fonction helper pour découvrir les fichiers CSV dans les répertoires spécifiés."""
    print("Recherche des fichiers CSV...")
    found_files = []
    
    for directory in search_dirs:
        if os.path.exists(directory):
            try:
                # Recherche récursive des fichiers CSV
                for root, dirs, files in os.walk(directory):
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if csv_files:
                        print(f"Trouvé {len(csv_files)} fichiers CSV dans {root}")
                        for csv_file in csv_files:
                            path = os.path.join(root, csv_file)
                            try:
                                # Obtenir les infos de base du fichier
                                size = os.path.getsize(path) / 1024  # KB
                                found_files.append({
                                    'path': path,
                                    'size': f"{size:.1f} KB",
                                    'directory': root
                                })
                            except Exception as e:
                                print(f"Erreur lors de la lecture des infos pour {path}: {str(e)}")
            except PermissionError:
                print(f"Permission refusée pour le répertoire: {directory}")
    
    if not found_files:
        return """Aucun fichier CSV trouvé dans l'espace de travail.

Suggestions:
1. Vérifie que les fichiers CSV existent dans l'espace de travail
2. Vérifie les permissions des fichiers
3. Assure-toi que les fichiers ont l'extension .csv

Répertoires recherchés:
- data/csv_temp/
- ./data/csv_temp/"""

    # Formater la réponse
    response = ["Voici les fichiers CSV disponibles:"]
    for file in found_files:
        response.append(f"\n• {os.path.basename(file['path'])}")
        response.append(f"  Emplacement: {file['directory']}")
        response.append(f"  Taille: {file['size']}")
    
    response.append("\nPour charger un fichier spécifique, utilise: data_loader('nom_fichier.csv')")
    print(f"Découverte terminée: {len(found_files)} fichiers trouvés")
    return "\n".join(response)


def _load_file(file_context: str, search_dirs: list) -> str:
    """Fonction helper pour charger un fichier CSV spécifique."""
    print(f"Chargement du fichier: {file_context}")
    possible_paths = []
    
    # Si file_context ressemble à un nom de fichier, essayer différents emplacements
    if file_context.endswith('.csv'):
        possible_paths = [
            file_context,  # Chemin direct
            f"data/{file_context}",  # dossier data
            f"available/{file_context}",  # dossier available  
            f"./data/{file_context}",  # dossier data relatif
            f"./{file_context}",  # répertoire courant
        ]
        
        # Rechercher aussi récursivement le nom de fichier exact
        for directory in search_dirs:
            if os.path.exists(directory):
                try:
                    for root, dirs, files in os.walk(directory):
                        if file_context in files:
                            possible_paths.append(os.path.join(root, file_context))
                except PermissionError:
                    continue
    else:
        # Rechercher les fichiers correspondants dans tous les répertoires
        for directory in search_dirs:
            if os.path.exists(directory):
                try:
                    for root, dirs, files in os.walk(directory):
                        csv_files = [f for f in files if f.endswith('.csv')]
                        for csv_file in csv_files:
                            if file_context.lower() in csv_file.lower():
                                possible_paths.append(os.path.join(root, csv_file))
                except PermissionError:
                    continue
    
    print(f"Tentative de chargement depuis {len(possible_paths)} emplacements possibles...")
    
    # Essayer de charger le premier fichier CSV disponible
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Chargement du CSV depuis: {path}")
                df = pd.read_csv(path)
                print(f"Chargement réussi: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                return _generate_data_summary(df, path)
        except Exception as e:
            print(f"Erreur lors du chargement de {path}: {str(e)}")
            continue
    
    # Si aucun fichier trouvé, passer en mode découverte
    print("Aucun fichier correspondant trouvé, basculement en mode découverte")
    return _discover_files(search_dirs)


def _generate_data_summary(df: pd.DataFrame, file_path: str) -> str:
    """Fonction helper pour générer un résumé complet des données."""
    try:
        # Conversion sécurisée en chaîne pour l'aperçu des données
        preview_str = df.head().to_string(max_cols=10, max_rows=5)
        
        # Analyse sécurisée des valeurs manquantes
        missing_values = df.isnull().sum()
        missing_str = missing_values.to_string() if not missing_values.empty else "Aucune valeur manquante"
        
        # Analyse sécurisée des types de données
        dtypes_str = df.dtypes.to_string()
        
        # Résumé sécurisé des données numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_summary = df.describe().to_string(max_cols=10)
        else:
            numeric_summary = 'Aucune colonne numérique trouvée'
        
        return f"""Parfait ! J'ai chargé ton fichier CSV '{file_path}' avec succès.

Voici ce que contiennent tes données:
• Nombre de lignes: {df.shape[0]}
• Nombre de colonnes: {df.shape[1]}
• Taille en mémoire: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
• Colonnes disponibles: {list(df.columns)}

Aperçu des premières lignes:
{preview_str}

Valeurs manquantes:
{missing_str}

Types de données:
{dtypes_str}

Résumé statistique des colonnes numériques:
{numeric_summary}

Pour utiliser ces données en Python:
df = pd.read_csv('{file_path}')

Rappel important: Après avoir créé un graphique, n'oublie pas d'utiliser display_figures() pour l'afficher !

Que souhaites-tu analyser dans ces données ?"""

    except Exception as e:
        error_msg = f"Erreur lors de la génération du résumé: {str(e)}"
        print(error_msg)
        return f"""J'ai chargé ton fichier CSV '{file_path}' mais il y a eu un problème pour générer le résumé complet.

Informations de base:
• {df.shape[0]} lignes, {df.shape[1]} colonnes
• Colonnes: {list(df.columns)}

Erreur: {error_msg}

Tu peux quand même utiliser tes données avec:
df = pd.read_csv('{file_path}')"""


@tool
def display_figures(figures_dict: Dict[str, Any], figure_type: str = "auto") -> str:
    """
    Outil d'affichage unifié pour les graphiques matplotlib et plotly.
    
    Affiche en sécurité plusieurs graphiques tout en gérant la mémoire et en fournissant
    des retours détaillés. À appeler IMMÉDIATEMENT après avoir créé un graphique.
    
    Args:
        figures_dict: Dictionnaire associant les noms de graphiques aux objets graphiques.
                     Maximum 10 graphiques par appel pour éviter les problèmes de mémoire.
                     Utilise des noms descriptifs: {"tendance_ventes": fig1, "correlation": fig2}
                     
        figure_type: Type de graphiques à afficher:
                    - "matplotlib": Pour les graphiques matplotlib/seaborn
                    - "plotly": Pour les graphiques plotly
                    - "auto": (défaut) Détecte automatiquement le type de graphique
    
    Returns:
        Message de statut sur les graphiques affichés avec succès ou informations d'erreur
        spécifiques avec conseils de dépannage.
    """
    try:
        print(f"Affichage de {len(figures_dict) if isinstance(figures_dict, dict) else 'invalide'} graphiques")
        
        # Valider l'entrée
        if not isinstance(figures_dict, dict):
            error_msg = f"""Je m'attendais à recevoir un dictionnaire de graphiques, mais j'ai reçu: {type(figures_dict)}

Usage correct:
1. Pour matplotlib:
   fig, ax = plt.subplots()
   ax.plot(data)
   display_figures({{"nom_graphique": fig}})

2. Pour seaborn:
   plot = sns.histplot(data)
   fig = plot.get_figure()
   display_figures({{"nom_graphique": fig}})

3. Pour plotly:
   fig = go.Figure()
   display_figures({{"nom_graphique": fig}})

Attention: Ne passe pas la librairie elle-même (plt, sns, ou go)"""
            print(error_msg)
            return error_msg
        
        if not figures_dict:
            warning_msg = """Attention: Dictionnaire de graphiques vide.

Assure-toi de créer des graphiques avant de les afficher. Exemple:
fig, ax = plt.subplots()
ax.plot(data)
display_figures({{"nom_graphique": fig}})"""
            print(warning_msg)
            return warning_msg
        
        # Vérifier la limite de graphiques
        max_figures = VISUALIZATION_CONFIG.get("max_figures_per_call", 10)
        if len(figures_dict) > max_figures:
            error_msg = f"""Trop de graphiques ({len(figures_dict)}). Maximum {max_figures} autorisés par appel.

Solution: Divise en plusieurs appels avec ≤{max_figures} graphiques chacun. Exemple:
# Premier appel
display_figures({{"graphique1": fig1, "graphique2": fig2}})

# Deuxième appel
display_figures({{"graphique3": fig3, "graphique4": fig4}})"""
            print(error_msg)
            return error_msg
        
        displayed_count = 0
        failed_figures = []
        
        for fig_name, fig_obj in figures_dict.items():
            try:
                print(f"Traitement du graphique: {fig_name} (type: {type(fig_obj)})")
                
                # Déterminer le type de graphique si auto
                current_figure_type = figure_type
                if current_figure_type == "auto":
                    if isinstance(fig_obj, go.Figure):
                        current_figure_type = "plotly"
                        print(f"Détection automatique: graphique plotly")
                    else:
                        current_figure_type = "matplotlib"
                        print(f"Détection automatique: graphique matplotlib")
                
                # Gérer les graphiques matplotlib
                if current_figure_type == "matplotlib":
                    # Gérer les objets axes en récupérant leur figure
                    if hasattr(fig_obj, 'get_figure'):
                        print(f"Conversion axes vers figure pour: {fig_name}")
                        fig_obj = fig_obj.get_figure()
                    
                    if hasattr(fig_obj, 'savefig'):
                        st.pyplot(fig_obj)
                        plt.close(fig_obj)  # Fermer pour libérer la mémoire
                        displayed_count += 1
                        print(f"Graphique matplotlib affiché avec succès: {fig_name}")
                    else:
                        failed_figures.append(f"""{fig_name}: Ce n'est pas un objet graphique matplotlib valide (type: {type(fig_obj)})

Vérifications:
1. Assure-toi de passer l'objet figure, pas la librairie:
   fig, ax = plt.subplots()  # Correct
   display_figures({{"nom": fig}})  # Correct
   display_figures({{"nom": plt}})  # Incorrect

2. Pour les graphiques seaborn, récupère l'objet figure:
   plot = sns.histplot(data)
   fig = plot.get_figure()  # Important!
   display_figures({{"nom": fig}})""")
                
                # Gérer les graphiques plotly
                elif current_figure_type == "plotly":
                    if isinstance(fig_obj, go.Figure):
                        st.plotly_chart(fig_obj, use_container_width=True)
                        displayed_count += 1
                        print(f"Graphique plotly affiché avec succès: {fig_name}")
                    else:
                        failed_figures.append(f"""{fig_name}: Ce n'est pas un objet graphique plotly valide (type: {type(fig_obj)})

Vérifications:
1. Assure-toi de passer un objet go.Figure:
   fig = go.Figure()  # Correct
   display_figures({{"nom": fig}})  # Correct
   display_figures({{"nom": go}})   # Incorrect""")
                
                else:
                    failed_figures.append(f"""{fig_name}: Type de graphique invalide '{current_figure_type}'

Types valides:
- "matplotlib": Pour les graphiques matplotlib/seaborn
- "plotly": Pour les graphiques plotly
- "auto": (défaut) Détection automatique du type""")
                    
            except Exception as e:
                error_details = f"""{fig_name}: {str(e)}

Vérifications:
1. Ton objet graphique est-il valide ?
2. Passes-tu bien l'objet figure, pas la librairie ?
3. Pour seaborn, utilise .get_figure()
4. Pour plotly, assure-toi de passer une instance go.Figure
5. Type d'erreur: {type(e).__name__}"""
                failed_figures.append(error_details)
                print(f"Erreur lors de l'affichage de {fig_name}: {str(e)}")
                st.error(f"Erreur lors de l'affichage du graphique {fig_name}: {str(e)}")
        
        # Préparer le message de résultat
        result_parts = [f"Résultats de l'affichage:"]
        result_parts.append(f"✓ {displayed_count} graphique(s) affiché(s) avec succès")
        
        if failed_figures:
            result_parts.append(f"✗ {len(failed_figures)} graphique(s) n'ont pas pu être affichés")
            result_parts.append("Détails des échecs:")
            for failure in failed_figures:
                result_parts.append(f"   - {failure}")
        
        if displayed_count > 0:
            result_parts.append(f"\nMémoire libérée pour {displayed_count} graphique(s)")
        
        final_result = "\n".join(result_parts)
        print(f"Affichage terminé: {displayed_count} réussis, {len(failed_figures)} échoués")
        return final_result
        
    except Exception as e:
        error_msg = f"Erreur critique lors de l'affichage: {str(e)}"
        print(error_msg)
        return error_msg + "\n\nConseil: Vérifie que tu passes des objets graphiques valides dans un dictionnaire" 