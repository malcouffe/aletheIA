import pandas as pd
import os
import csv
import chardet
from typing import List, Dict, Union, Optional, Tuple, Any


class CSVProcessor:
    """
    Classe pour traiter et valider les fichiers CSV.
    """

    def __init__(self, encoding: str = "utf-8", separator: str = ","):
        """
        Initialise le processeur CSV.
        
        Args:
            encoding: Encodage du fichier CSV
            separator: Séparateur de colonnes
        """
        self.encoding = encoding
        self.separator = separator
        
    def check_file_exists(self, file_path: str) -> bool:
        """
        Vérifie si le fichier existe.
        
        Args:
            file_path: Chemin du fichier CSV
            
        Returns:
            bool: True si le fichier existe, False sinon
        """
        return os.path.isfile(file_path)
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Détecte l'encodage du fichier CSV.
        
        Args:
            file_path: Chemin du fichier CSV
            
        Returns:
            str: Encodage détecté
        """
        # Lire un échantillon du fichier pour détecter l'encodage
        with open(file_path, 'rb') as f:
            sample = f.read(10000)  # Lire les premiers 10000 octets
            
        result = chardet.detect(sample)
        detected_encoding = result['encoding']
        
        if detected_encoding is None:
            return self.encoding  # Revenir à l'encodage par défaut
            
        return detected_encoding
    
    def detect_separator(self, file_path: str, encoding: str = None) -> str:
        """
        Détecte automatiquement le séparateur du fichier CSV.
        
        Args:
            file_path: Chemin du fichier CSV
            encoding: Encodage du fichier (si None, utilise self.encoding)
            
        Returns:
            str: Séparateur détecté
        """
        if encoding is None:
            encoding = self.encoding
            
        # Liste des séparateurs courants à tester
        separators = [',', ';', '\t', '|', ' ']
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                sample = f.read(5000)  # Lire les premières lignes
                
            # Compter le nombre d'occurrences de chaque séparateur
            separator_counts = {sep: sample.count(sep) for sep in separators}
            
            # Vérifier si le dialect sniffer peut détecter le séparateur
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=''.join(separators))
                detected_sep = dialect.delimiter
                if detected_sep in separators:
                    return detected_sep
            except:
                pass  # Si le sniffer échoue, on continue avec la méthode par comptage
                
            # Sinon, prendre le séparateur le plus fréquent
            if max(separator_counts.values()) > 0:
                return max(separator_counts.items(), key=lambda x: x[1])[0]
                
        except Exception as e:
            print(f"Erreur lors de la détection du séparateur: {e}")
            
        # Si toutes les méthodes échouent, revenir au séparateur par défaut
        return self.separator
    
    def validate_csv(self, file_path: str, auto_detect: bool = True) -> Tuple[bool, str]:
        """
        Valide la structure et le contenu du fichier CSV.
        
        Args:
            file_path: Chemin du fichier CSV
            auto_detect: Si True, tente de détecter automatiquement l'encodage et le séparateur
            
        Returns:
            Tuple[bool, str]: (est_valide, message_erreur)
        """
        if not self.check_file_exists(file_path):
            return False, "Le fichier n'existe pas"
            
        encoding = self.encoding
        separator = self.separator
        
        if auto_detect:
            try:
                # Détecter l'encodage
                detected_encoding = self.detect_encoding(file_path)
                if detected_encoding:
                    encoding = detected_encoding
                    
                # Détecter le séparateur
                separator = self.detect_separator(file_path, encoding)
            except Exception as e:
                print(f"Erreur lors de la détection automatique: {str(e)}")
                # Continuer avec les valeurs par défaut
        
        try:
            # Tente de lire le CSV
            df = pd.read_csv(file_path, sep=separator, encoding=encoding, 
                            nrows=5, on_bad_lines='warn', low_memory=False)
            
            # Vérifications basiques
            if len(df.columns) < 1:
                return False, "Le fichier ne contient pas de colonnes"
            
            # Vérifier si toutes les colonnes sont "Unnamed"
            unnamed_cols = sum(1 for col in df.columns if 'Unnamed' in str(col))
            if unnamed_cols == len(df.columns):
                # Essayer un autre séparateur
                for sep in [',', ';', '\t', '|']:
                    if sep != separator:
                        try:
                            test_df = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=2)
                            if len(test_df.columns) > unnamed_cols:
                                return False, f"Le séparateur semble être '{sep}' plutôt que '{separator}'"
                        except:
                            pass
                            
            # Vérifier les valeurs manquantes
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                return True, f"Le fichier est valide mais contient {missing_count} valeurs manquantes dans les 5 premières lignes"
                
            return True, "Le fichier CSV est valide"
            
        except UnicodeDecodeError:
            # Si l'encodage spécifié échoue, essayer de détecter automatiquement
            try:
                detected_encoding = self.detect_encoding(file_path)
                if detected_encoding and detected_encoding != encoding:
                    try:
                        # Réessayer avec l'encodage détecté
                        df = pd.read_csv(file_path, sep=separator, encoding=detected_encoding, nrows=5)
                        return True, f"Le fichier est valide mais utilise l'encodage {detected_encoding}"
                    except:
                        pass
                        
                return False, f"Problème d'encodage. L'encodage {encoding} ne semble pas être correct."
            except:
                return False, "Impossible de déterminer l'encodage correct"
                
        except pd.errors.ParserError as e:
            return False, f"Erreur lors du parsing du CSV. Format incorrect: {str(e)}"
        except pd.errors.EmptyDataError:
            return False, "Le fichier CSV est vide"
        except Exception as e:
            return False, f"Erreur lors de la validation: {str(e)}"
    
    def read_csv(self, file_path: str, limit: Optional[int] = None, auto_detect: bool = True) -> Optional[pd.DataFrame]:
        """
        Lit le fichier CSV et retourne un DataFrame pandas.
        
        Args:
            file_path: Chemin du fichier CSV
            limit: Nombre maximum de lignes à lire (None pour tout lire)
            auto_detect: Si True, tente de détecter automatiquement l'encodage et le séparateur
            
        Returns:
            Optional[pd.DataFrame]: DataFrame contenant les données ou None en cas d'erreur
        """
        if auto_detect:
            # Auto-détection de l'encodage et du séparateur
            try:
                encoding = self.detect_encoding(file_path)
                separator = self.detect_separator(file_path, encoding)
            except Exception as e:
                print(f"Erreur lors de la détection automatique: {str(e)}")
                encoding = self.encoding
                separator = self.separator
        else:
            encoding = self.encoding
            separator = self.separator
                
        try:
            if limit:
                return pd.read_csv(file_path, sep=separator, encoding=encoding, 
                                  nrows=limit, on_bad_lines='warn', low_memory=False)
            else:
                return pd.read_csv(file_path, sep=separator, encoding=encoding, 
                                  on_bad_lines='warn', low_memory=False)
        except Exception as e:
            print(f"Erreur lors de la lecture du CSV: {str(e)}")
            
            # Essayer diverses combinaisons d'encodage et de séparateurs
            for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
                for sep in [",", ";", "\t", "|"]:
                    try:
                        if limit:
                            df = pd.read_csv(file_path, sep=sep, encoding=enc, 
                                            nrows=limit, on_bad_lines='warn')
                        else:
                            df = pd.read_csv(file_path, sep=sep, encoding=enc, 
                                            on_bad_lines='warn')
                        print(f"Réussite avec encoding={enc}, separator={sep}")
                        self.encoding = enc
                        self.separator = sep
                        return df
                    except:
                        continue
                
            return None
    
    def get_columns(self, file_path: str, auto_detect: bool = True) -> List[str]:
        """
        Retourne la liste des colonnes du CSV.
        
        Args:
            file_path: Chemin du fichier CSV
            auto_detect: Si True, tente de détecter automatiquement l'encodage et le séparateur
            
        Returns:
            List[str]: Liste des noms de colonnes
        """
        df = self.read_csv(file_path, limit=1, auto_detect=auto_detect)
        if df is not None:
            return df.columns.tolist()
        return []
        
    def analyze_csv(self, file_path: str, auto_detect: bool = True) -> Dict[str, Any]:
        """
        Analyse le fichier CSV et retourne des statistiques basiques.
        
        Args:
            file_path: Chemin du fichier CSV
            auto_detect: Si True, tente de détecter automatiquement l'encodage et le séparateur
            
        Returns:
            Dict[str, Any]: Informations sur le CSV
        """
        df = self.read_csv(file_path, auto_detect=auto_detect)
        if df is None:
            return {}
            
        # Statistiques de base
        result = {
            "nombre_lignes": len(df),
            "nombre_colonnes": len(df.columns),
            "colonnes": df.columns.tolist(),
            "types_colonnes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "valeurs_manquantes": df.isnull().sum().to_dict(),
            "aperçu": df.head(5).to_dict()
        }
        
        # Statistiques avancées
        try:
            # Informations sur la mémoire occupée
            result["memoire_usage"] = {
                "total": f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB",
                "par_colonne": {col: f"{mem / 1024:.2f} KB" for col, mem in 
                              df.memory_usage(deep=True).drop('Index').items()}
            }
            
            # Statistiques par type de colonne
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Stockage du nombre de colonnes par type
            result["nombre_colonnes_par_type"] = {
                "numeriques": len(numeric_cols),
                "categorielles": len(categorical_cols),
                "dates": len(date_cols)
            }
            
            # Statistiques sur les valeurs uniques
            result["valeurs_uniques"] = {col: int(df[col].nunique()) for col in df.columns}
            
            # Descriptions statistiques pour les colonnes numériques
            if numeric_cols:
                result["statistiques_numeriques"] = df[numeric_cols].describe().to_dict()
                
            # Distribution des valeurs pour les colonnes catégorielles (top 5)
            if categorical_cols:
                result["statistiques_categorielles"] = {}
                for col in categorical_cols[:5]:  # Limiter à 5 colonnes
                    try:
                        result["statistiques_categorielles"][col] = df[col].value_counts().head(5).to_dict()
                    except:
                        continue
                        
            # Détection des corrélations entre variables numériques
            if len(numeric_cols) > 1:
                try:
                    result["correlations"] = df[numeric_cols].corr().to_dict()
                except:
                    pass
                    
        except Exception as e:
            # Les statistiques avancées échouent, on les ignore
            print(f"Erreur lors du calcul des statistiques avancées: {str(e)}")
        
        return result
