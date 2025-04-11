from smolagents import Tool
import tempfile
import os
import gc  # Pour le garbage collector
from typing import Dict, Any, Optional
from utils.pipeline_indexation.csv_processor import CSVProcessor

class CSVAnalyzerTool(Tool):
    name = "csv_analyzer"
    description = "Analyzes CSV files to provide insights about their structure and content."
    inputs = {
        "file_content": {
            "type": "string",
            "description": "The content of the CSV file as text",
        },
        "separator": {
            "type": "string", 
            "description": "CSV separator character (default is ',')",
            "default": ",",
            "nullable": True
        },
        "encoding": {
            "type": "string",
            "description": "File encoding (default is 'utf-8')",
            "default": "utf-8",
            "nullable": True
        },
        "chunk_size": {
            "type": "integer",
            "description": "Size of chunks to process for large files (to limit memory usage)",
            "default": 100000,
            "nullable": True
        },
        "figures_dir": {
            "type": "string",
            "description": "Directory where to save generated figures",
            "default": "./figures",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, file_content: str, separator: str = ",", encoding: str = "utf-8", 
               chunk_size: Optional[int] = 100000, figures_dir: str = "./figures") -> str:
        """
        Analyze the CSV file content and return insights.
        
        Args:
            file_content: The content of the CSV file as text
            separator: CSV separator character
            encoding: File encoding
            chunk_size: Size of chunks to process for large files (None for no chunking)
            figures_dir: Directory where to save generated figures
            
        Returns:
            Analysis results as a formatted string
        """
        # S'assurer que le dossier des figures existe
        os.makedirs(figures_dir, exist_ok=True)
        
        # Sauvegarder le contenu dans un fichier temporaire
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_content.encode(encoding))
            
            # Utiliser le CSVProcessor existant pour analyser le fichier
            processor = CSVProcessor(encoding=encoding, separator=separator)
            
            # Vérifier si le fichier est valide
            is_valid, message = processor.validate_csv(temp_path)
            if not is_valid:
                return f"Erreur dans le fichier CSV: {message}"
            
            # Analyser le fichier en tenant compte de la limite mémoire
            process_options = {}
            if chunk_size is not None:
                process_options["chunk_size"] = chunk_size
                
            # Analyse complète du CSV avec gestion optimisée de la mémoire pour les gros fichiers
            analysis = processor.analyze_csv(temp_path, auto_detect=True)
            
            # Libérer la mémoire après l'analyse
            gc.collect()
            
            # Formater les résultats
            result = "## Analyse du fichier CSV\n\n"
            result += f"- **Nombre de lignes**: {analysis.get('nombre_lignes', 'N/A')}\n"
            result += f"- **Nombre de colonnes**: {analysis.get('nombre_colonnes', 'N/A')}\n"
            
            if analysis.get("memoire_usage"):
                result += f"- **Utilisation mémoire totale**: {analysis['memoire_usage'].get('total', 'N/A')}\n"
            
            # Liste des colonnes avec leurs types
            result += "\n### Colonnes et types de données\n"
            if 'types_colonnes' in analysis:
                for col, dtype in analysis['types_colonnes'].items():
                    result += f"- `{col}`: {dtype}\n"
            
            # Valeurs manquantes
            result += "\n### Valeurs manquantes\n"
            if 'valeurs_manquantes' in analysis:
                total_missing = sum(analysis['valeurs_manquantes'].values())
                result += f"- **Total de valeurs manquantes**: {total_missing}\n"
                result += "- **Détail par colonne**:\n"
                for col, count in analysis['valeurs_manquantes'].items():
                    if count > 0:
                        result += f"  - `{col}`: {count} valeurs manquantes\n"
            
            # Statistiques avancées si disponibles
            if 'statistiques_numeriques' in analysis:
                result += "\n### Statistiques descriptives\n"
                result += "Pour les colonnes numériques, voici les principales statistiques:\n\n"
                for col, stats in analysis['statistiques_numeriques'].items():
                    result += f"#### `{col}`\n"
                    for stat_name, value in stats.items():
                        result += f"- {stat_name}: {value}\n"
                    result += "\n"
            
            # Distribution des valeurs pour colonnes catégorielles
            if 'statistiques_categorielles' in analysis:
                result += "\n### Distributions des catégories (Top 5)\n"
                for col, dist in analysis['statistiques_categorielles'].items():
                    result += f"#### `{col}`\n"
                    for val, count in dist.items():
                        result += f"- {val}: {count}\n"
                    result += "\n"
            
            return result
            
        except MemoryError:
            gc.collect()  # Essayer de libérer la mémoire
            if chunk_size is not None:
                return f"Erreur mémoire: Le fichier est trop volumineux pour être traité avec la limite de chunks actuelle ({chunk_size}). Réduisez la taille du fichier ou augmentez les ressources système."
            else:
                return f"Erreur mémoire: Le fichier est trop volumineux. Essayez d'activer l'option 'Limiter l'utilisation de la mémoire'."
        
        except Exception as e:
            return f"Erreur lors de l'analyse du CSV: {str(e)}"
        
        finally:
            # Nettoyer le fichier temporaire
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
            # Forcer le garbage collection pour libérer la mémoire
            gc.collect()
