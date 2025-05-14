"""
Gestionnaire de base de données vectorielle (ChromaDB)
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from managed_agent.vector_config import EMBEDDING_CONFIG

class VectorDBManager:
    """
    Classe utilitaire pour gérer les connexions à ChromaDB et diagnostiquer les problèmes.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialise le gestionnaire de base de données vectorielle.
        
        Args:
            db_path: Chemin vers la base de données ChromaDB
        """
        self.db_path = db_path
        self.embedding_function = None
        self.chroma_client = None
        self.vectordb = None
        self.collection_name = None
        
        if db_path:
            self.initialize(db_path)
    
    def initialize(self, db_path: str) -> bool:
        """
        Initialise la connexion à la base de données.
        
        Args:
            db_path: Chemin vers la base de données ChromaDB
            
        Returns:
            bool: True si l'initialisation a réussi, False sinon
        """
        self.db_path = db_path
        
        # Vérifier que le répertoire existe
        if not os.path.exists(db_path):
            print(f"Database directory does not exist: {db_path}")
            return False
        
        # Initialiser l'embedding function
        try:
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["model_name"],
                model_kwargs=EMBEDDING_CONFIG["model_kwargs"],
                encode_kwargs=EMBEDDING_CONFIG["encode_kwargs"]
            )
            
            # Initialiser le client Chroma
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Vérifier les collections disponibles
            collections = self.chroma_client.list_collections()
            if not collections:
                print(f"No collections found in the database at {db_path}")
                return False
            
            # Utiliser la première collection disponible
            self.collection_name = collections[0].name
            print(f"Found collection: {self.collection_name}")
            
            # Initialiser le vectorstore
            self.vectordb = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Test de la connexion
            test_results = self.vectordb.similarity_search("test connection", k=1)
            print(f"Database connection test successful: found {len(test_results)} results")
            
            return True
        
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            return False
    
    def get_vectordb(self) -> Optional[Chroma]:
        """
        Retourne l'instance de VectorStore initialisée.
        
        Returns:
            Optional[Chroma]: L'instance de VectorStore ou None si non initialisée
        """
        return self.vectordb
    
    def check_database_integrity(self) -> Tuple[bool, str]:
        """
        Vérifie l'intégrité de la base de données SQLite.
        
        Returns:
            Tuple[bool, str]: (True/False, message descriptif)
        """
        if not self.db_path or not os.path.exists(self.db_path):
            return False, f"Database path does not exist: {self.db_path}"
        
        # Chercher les fichiers SQLite
        try:
            all_files = os.listdir(self.db_path)
            sqlite_files = [f for f in all_files if f.endswith('.sqlite3')]
            
            if not sqlite_files:
                return False, f"No SQLite files found in {self.db_path}"
            
            # Tester chaque fichier SQLite
            for db_file in sqlite_files:
                db_path = os.path.join(self.db_path, db_file)
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check")
                    integrity = cursor.fetchone()[0]
                    conn.close()
                    
                    if integrity != "ok":
                        return False, f"Database integrity check failed: {integrity}"
                    
                except Exception as e:
                    return False, f"Error checking database integrity: {e}"
            
            return True, "Database integrity check passed"
        
        except Exception as e:
            return False, f"Error checking database files: {e}"
    
    def repair_database(self) -> Tuple[bool, str]:
        """
        Tente de réparer la base de données.
        
        Returns:
            Tuple[bool, str]: (True/False, message descriptif)
        """
        if not self.db_path or not os.path.exists(self.db_path):
            return False, f"Database path does not exist: {self.db_path}"
        
        try:
            # Créer un nouveau client et voir s'il est capable d'accéder à la base
            alt_client = chromadb.PersistentClient(path=self.db_path)
            collections = alt_client.list_collections()
            
            if not collections:
                return False, "No collections found, unable to repair"
            
            # Réinitialiser avec le nouveau client
            self.chroma_client = alt_client
            self.collection_name = collections[0].name
            
            self.vectordb = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Test de la connexion
            test_results = self.vectordb.similarity_search("test connection", k=1)
            
            return True, f"Database repaired, found {len(test_results)} results"
        
        except Exception as e:
            return False, f"Failed to repair database: {e}"
    
    def diagnose_error(self, error: Exception) -> str:
        """
        Effectue un diagnostic avancé des erreurs de base de données.
        
        Args:
            error: L'exception à diagnostiquer
            
        Returns:
            str: Rapport de diagnostic
        """
        diagnostic = ["Database Error Diagnostic:"]
        
        # Informations sur le chemin
        diagnostic.append(f"Database path: {self.db_path}")
        if self.db_path:
            diagnostic.append(f"Path exists: {os.path.exists(self.db_path)}")
        
        # Vérifier si c'est une erreur SQLite
        if "sqlite" in str(error).lower():
            diagnostic.append("SQLite error detected")
            
            if self.db_path and os.path.exists(self.db_path):
                try:
                    # Lister les fichiers
                    all_files = os.listdir(self.db_path)
                    diagnostic.append(f"Files in directory: {', '.join(all_files)}")
                    
                    # Chercher les fichiers SQLite
                    sqlite_files = [f for f in all_files if f.endswith('.sqlite3')]
                    diagnostic.append(f"SQLite files: {', '.join(sqlite_files)}")
                    
                    # Vérifier les permissions et l'intégrité
                    for db_file in sqlite_files:
                        db_path = os.path.join(self.db_path, db_file)
                        try:
                            # Permissions
                            import stat
                            st_mode = os.stat(db_path).st_mode
                            diagnostic.append(f"File permissions for {db_file}: {stat.filemode(st_mode)}")
                            
                            # Taille
                            size = os.path.getsize(db_path)
                            diagnostic.append(f"File size: {size} bytes")
                            
                            # Intégrité
                            try:
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                                cursor.execute("PRAGMA integrity_check")
                                integrity = cursor.fetchone()[0]
                                conn.close()
                                diagnostic.append(f"Database integrity: {integrity}")
                            except Exception as sql_e:
                                diagnostic.append(f"Error checking integrity: {sql_e}")
                                
                        except Exception as file_e:
                            diagnostic.append(f"Error accessing file {db_file}: {file_e}")
                
                except Exception as dir_e:
                    diagnostic.append(f"Error accessing directory: {dir_e}")
        
        # Ajouter l'erreur originale
        diagnostic.append(f"Original error: {str(error)}")
        
        # Retourner le rapport complet
        return "\n".join(diagnostic)


# Fonction utilitaire pour obtenir une instance VectorDBManager
def get_vector_db_manager(db_path: str) -> VectorDBManager:
    """
    Crée et initialise un gestionnaire de base de données vectorielle.
    
    Args:
        db_path: Chemin vers la base de données
        
    Returns:
        VectorDBManager: Un gestionnaire initialisé
    """
    manager = VectorDBManager()
    manager.initialize(db_path)
    return manager 