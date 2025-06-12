"""
File management functionality including cleanup and deletion.
"""
import streamlit as st
import os
import shutil
from core.session_manager import save_persistent_state


def cleanup_single_file_resources(file_id: str, details: dict):
    """Attempts to clean up resources associated with a single file (temp files, DBs, figures, images, tables, exports)."""
    cleanup_summary = {
        "temp_files": 0,
        "vector_db": 0,
        "figures": 0,
        "images": 0,
        "tables": 0,
        "exports": 0,
        "errors": []
    }
    
    # 1. Nettoyer les fichiers temporaires
    temp_path = details.get('temp_path')
    if temp_path and os.path.exists(temp_path):
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
                cleanup_summary["temp_files"] += 1
                print(f"Removed temp file: {temp_path}")
                temp_dir = os.path.dirname(temp_path)
                if os.path.basename(temp_dir) == file_id:
                     try:
                         os.rmdir(temp_dir)
                         print(f"Removed temp directory: {temp_dir}")
                         session_dir = os.path.dirname(temp_dir)
                         if not os.listdir(session_dir) and session_dir != os.path.join("data", "temp_files"):
                             os.rmdir(session_dir)
                             print(f"Removed session temp directory: {session_dir}")
                     except OSError as e:
                         cleanup_summary["errors"].append(f"Could not remove directory {temp_dir}: {e}")
            elif os.path.isdir(temp_path): 
                shutil.rmtree(temp_path)
                cleanup_summary["temp_files"] += 1
                print(f"Removed temp directory (shutil): {temp_path}")
        except Exception as e:
            cleanup_summary["errors"].append(f"Error removing temp resource {temp_path}: {e}")

    # 2. Nettoyer la base vectorielle
    db_path = details.get('db_path')
    if db_path and os.path.exists(db_path):
        try:
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
                cleanup_summary["vector_db"] += 1
                print(f"Removed vector DB directory: {db_path}")
                
                # Nettoyer les dossiers parents si vides
                class_dir = os.path.dirname(db_path)
                try:
                    if not os.listdir(class_dir):
                        os.rmdir(class_dir)
                        print(f"Removed classification DB directory: {class_dir}")
                        session_db_dir = os.path.dirname(class_dir)
                        if not os.listdir(session_db_dir) and "vectordb" in session_db_dir:
                            os.rmdir(session_db_dir)
                            print(f"Removed session DB directory: {session_db_dir}")
                except OSError as e:
                    cleanup_summary["errors"].append(f"Could not remove DB directory {class_dir}: {e}")
            else:
                 cleanup_summary["errors"].append(f"Expected DB path {db_path} to be a directory, but it is not")
        except Exception as e:
            cleanup_summary["errors"].append(f"Error removing vector DB {db_path}: {e}")

    # 3. Nettoyer les figures/graphiques
    try:
        session_id = st.session_state.session_id
        figures_dir = os.path.join('data', 'figures', session_id, file_id)
        if os.path.exists(figures_dir) and os.path.isdir(figures_dir):
            # Compter les fichiers avant suppression
            figures_count = sum(1 for f in os.listdir(figures_dir) if os.path.isfile(os.path.join(figures_dir, f)))
            cleanup_summary["figures"] += figures_count
            
            shutil.rmtree(figures_dir)
            print(f"Removed figures directory: {figures_dir} ({figures_count} files)")
            
            # Nettoyer le dossier parent si vide
            session_figures_dir = os.path.dirname(figures_dir)
            try:
                if not os.listdir(session_figures_dir) and session_figures_dir != os.path.join('data', 'figures'):
                    os.rmdir(session_figures_dir)
                    print(f"Removed session figures directory: {session_figures_dir}")
            except OSError as e:
                 cleanup_summary["errors"].append(f"Could not remove session figures directory {session_figures_dir}: {e}")
    except AttributeError:
         cleanup_summary["errors"].append(f"Could not clean up figures for {file_id}: session_id not found")
    except Exception as e:
        cleanup_summary["errors"].append(f"Error removing figures directory for {file_id}: {e}")

    # 4. Nettoyer les images extraites (pour les PDFs)
    images_dir = os.path.join("data", "output", "images")
    if os.path.exists(images_dir):
        try:
            # Chercher les images associées à ce fichier (basé sur le pattern filename)
            filename = details.get('filename', '')
            if filename:
                base_name = filename.replace('.pdf', '').replace('.csv', '')
                for img_file in os.listdir(images_dir):
                    if base_name in img_file or file_id in img_file:
                        img_path = os.path.join(images_dir, img_file)
                        if os.path.isfile(img_path):
                            os.remove(img_path)
                            cleanup_summary["images"] += 1
                            print(f"Removed extracted image: {img_path}")
        except Exception as e:
            cleanup_summary["errors"].append(f"Error cleaning extracted images: {e}")

    # 5. Nettoyer les tableaux extraits (pour les PDFs)
    tables_dir = os.path.join("data", "output", "tables")
    if os.path.exists(tables_dir):
        try:
            # Chercher les tableaux associés à ce fichier
            filename = details.get('filename', '')
            if filename:
                base_name = filename.replace('.pdf', '').replace('.csv', '')
                for table_file in os.listdir(tables_dir):
                    if base_name in table_file or file_id in table_file:
                        table_path = os.path.join(tables_dir, table_file)
                        if os.path.isfile(table_path):
                            os.remove(table_path)
                            cleanup_summary["tables"] += 1
                            print(f"Removed extracted table: {table_path}")
        except Exception as e:
            cleanup_summary["errors"].append(f"Error cleaning extracted tables: {e}")

    # 6. Nettoyer les exports (pour les PDFs)
    export_dir = os.path.join("data", "output")
    if os.path.exists(export_dir):
        try:
            filename = details.get('filename', '')
            if filename:
                base_name = filename.replace('.pdf', '').replace('.csv', '')
                # Chercher les dossiers d'export qui correspondent à ce fichier
                for item in os.listdir(export_dir):
                    item_path = os.path.join(export_dir, item)
                    if os.path.isdir(item_path) and base_name in item:
                        # Compter les fichiers dans l'export avant suppression
                        export_files = 0
                        for root, dirs, files in os.walk(item_path):
                            export_files += len(files)
                        
                        shutil.rmtree(item_path)
                        cleanup_summary["exports"] += export_files
                        print(f"Removed export directory: {item_path} ({export_files} files)")
        except Exception as e:
            cleanup_summary["errors"].append(f"Error cleaning export directories: {e}")

    # 7. Nettoyer les fichiers CSV temporaires spécifiques
    if details.get('type') == 'csv':
        csv_args = details.get('csv_args', {})
        csv_temp_path = csv_args.get('file_path')
        if csv_temp_path and os.path.exists(csv_temp_path):
            try:
                os.remove(csv_temp_path)
                cleanup_summary["temp_files"] += 1
                print(f"Removed CSV temp file: {csv_temp_path}")
                
                # Supprimer le dossier parent si vide
                csv_temp_dir = os.path.dirname(csv_temp_path)
                if os.path.exists(csv_temp_dir) and not os.listdir(csv_temp_dir):
                    os.rmdir(csv_temp_dir)
                    print(f"Removed CSV temp directory: {csv_temp_dir}")
            except Exception as e:
                cleanup_summary["errors"].append(f"Error removing CSV temp file {csv_temp_path}: {e}")

    # Afficher un résumé du nettoyage
    total_cleaned = sum([
        cleanup_summary["temp_files"],
        cleanup_summary["vector_db"],
        cleanup_summary["figures"],
        cleanup_summary["images"],
        cleanup_summary["tables"],
        cleanup_summary["exports"]
    ])
    
    if total_cleaned > 0:
        print(f"🧹 Cleanup summary for {file_id}:")
        print(f"  - Temp files: {cleanup_summary['temp_files']}")
        print(f"  - Vector DBs: {cleanup_summary['vector_db']}")
        print(f"  - Figures: {cleanup_summary['figures']}")
        print(f"  - Images: {cleanup_summary['images']}")
        print(f"  - Tables: {cleanup_summary['tables']}")
        print(f"  - Exports: {cleanup_summary['exports']}")
        print(f"  - Total files cleaned: {total_cleaned}")
    
    if cleanup_summary["errors"]:
        print(f"⚠️ Cleanup errors for {file_id}:")
        for error in cleanup_summary["errors"]:
            print(f"  - {error}")
    
    return cleanup_summary


def delete_file_callback(file_id_to_delete: str):
    """Callback function to handle file deletion."""
    if 'processed_files' in st.session_state and file_id_to_delete in st.session_state.processed_files:
        details = st.session_state.processed_files[file_id_to_delete]
        filename_for_message = details.get('filename', file_id_to_delete)
        
        # Effectuer le nettoyage complet
        cleanup_summary = cleanup_single_file_resources(file_id_to_delete, details)
        
        # Supprimer de la session
        del st.session_state.processed_files[file_id_to_delete]
        save_persistent_state(st.session_state.processed_files)
        
        # Créer un message informatif basé sur le nettoyage
        total_cleaned = sum([
            cleanup_summary["temp_files"],
            cleanup_summary["vector_db"], 
            cleanup_summary["figures"],
            cleanup_summary["images"],
            cleanup_summary["tables"],
            cleanup_summary["exports"]
        ])
        
        if total_cleaned > 0:
            cleanup_details = []
            if cleanup_summary["temp_files"] > 0:
                cleanup_details.append(f"{cleanup_summary['temp_files']} fichier(s) temporaire(s)")
            if cleanup_summary["vector_db"] > 0:
                cleanup_details.append("base vectorielle")
            if cleanup_summary["figures"] > 0:
                cleanup_details.append(f"{cleanup_summary['figures']} graphique(s)")
            if cleanup_summary["images"] > 0:
                cleanup_details.append(f"{cleanup_summary['images']} image(s) extraite(s)")
            if cleanup_summary["tables"] > 0:
                cleanup_details.append(f"{cleanup_summary['tables']} tableau(x) extrait(s)")
            if cleanup_summary["exports"] > 0:
                cleanup_details.append(f"{cleanup_summary['exports']} fichier(s) d'export")
            
            cleanup_text = ", ".join(cleanup_details)
            success_message = f"✅ Fichier '{filename_for_message}' supprimé avec {cleanup_text}."
        else:
            success_message = f"✅ Fichier '{filename_for_message}' supprimé (aucune ressource associée trouvée)."
        
        if cleanup_summary["errors"]:
            st.warning(f"⚠️ Suppression terminée avec {len(cleanup_summary['errors'])} avertissement(s). Voir la console pour les détails.")
        
        st.success(success_message)
        
        # Nettoyer la sélection si c'est le fichier sélectionné
        if st.session_state.get('selected_file_id_for_action') == file_id_to_delete:
            st.session_state.selected_file_id_for_action = None
        st.rerun()
    else:
        st.error(f"Impossible de supprimer le fichier : ID '{file_id_to_delete}' non trouvé.")


def select_pdf_for_action(file_id: str):
    """Sets the file_id for the PDF selected for classification/indexing."""
    st.session_state.selected_file_id_for_action = file_id 