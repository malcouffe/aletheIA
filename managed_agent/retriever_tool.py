import re
import os
from typing import List, Any, Optional, Dict, Tuple
from smolagents import Tool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from managed_agent.vector_config import SEARCH_CONFIG, DEFINITION_PREFIXES

# Variable globale pour stocker la référence à la base de données vectorielle
_GLOBAL_VECTORDB = None

# Configuration globale pour les recherches
_SEARCH_CONFIG = {
    "k_value": 5,
    "mmr_k": 5,
    "mmr_fetch_k": 10,
    "mmr_lambda_mult": 0.5,
    "max_docs": 7
}

# Mise à jour avec la configuration externe si disponible
if SEARCH_CONFIG:
    _SEARCH_CONFIG.update(SEARCH_CONFIG)

class RetrieverTool(Tool):
    """Outil pour rechercher des informations dans une base de données vectorielle via recherche sémantique."""
    
    name = "retriever"
    description = "Recherche des informations dans une base de données vectorielle en utilisant la similarité sémantique."
    inputs = {
        "query": {
            "type": "string",
            "description": "La requête à rechercher dans la base de données vectorielle",
            "nullable": True
        },
        "additional_notes": {
            "type": "string",
            "description": "Notes supplémentaires pour affiner la recherche",
            "nullable": True,
            "optional": True
        }
    }
    output_type = "string"
    
    def __init__(self, vectordb=None, **kwargs):
        # Initialisation de la classe parente
        super().__init__(**kwargs)
        
        # Stocker vectordb dans la variable globale
        if vectordb is not None:
            global _GLOBAL_VECTORDB
            _GLOBAL_VECTORDB = vectordb
            
            # Message de confirmation
            persist_dir = getattr(vectordb, '_persist_directory', 'Unknown location')
            print(f"RetrieverTool initialisé avec vectordb: {persist_dir}")
    
    def forward(self, query: str = "", additional_notes: str = None) -> str:
        """Méthode principale exécutée par Tool."""
        try:
            print(f"RetrieverTool: recherche pour '{query}'")
            
            # Accès à la variable globale
            global _GLOBAL_VECTORDB, _SEARCH_CONFIG
            
            # Vérification de la base de données
            if _GLOBAL_VECTORDB is None:
                print("ERREUR: Base de données vectorielle non initialisée")
                return "Erreur: Base de données vectorielle non initialisée"
            
            # 1. Nettoyage de la requête
            clean_query = self._preprocess_query(query)
            
            # 2. Recherche standard
            try:
                standard_results = _GLOBAL_VECTORDB.similarity_search(
                    clean_query,
                    k=_SEARCH_CONFIG['k_value']
                )
                print(f"Recherche standard: {len(standard_results)} documents trouvés")
            except Exception as e:
                print(f"Échec de la recherche standard: {e}")
                import traceback
                print(f"TRACEBACK: {traceback.format_exc()}")
                standard_results = []
            
            # 3. Recherche MMR pour diversité
            try:
                mmr_results = _GLOBAL_VECTORDB.max_marginal_relevance_search(
                    clean_query,
                    k=_SEARCH_CONFIG['mmr_k'],
                    fetch_k=_SEARCH_CONFIG['mmr_fetch_k'],
                    lambda_mult=_SEARCH_CONFIG['mmr_lambda_mult']
                )
                print(f"Recherche MMR: {len(mmr_results)} documents trouvés")
            except Exception as e:
                print(f"Échec de la recherche MMR: {e}")
                import traceback
                print(f"TRACEBACK: {traceback.format_exc()}")
                mmr_results = []
            
            # 4. Combiner et filtrer les résultats
            combined_results = self._combine_results(standard_results, mmr_results)
            
            # 5. Recherches alternatives si nécessaire
            if len(combined_results) < 2:
                print(f"Résultats insuffisants, essai de requêtes alternatives")
                alternative_results = self._try_alternative_queries(clean_query)
                combined_results.extend(alternative_results)
            
            # 6. Formatage et retour des résultats
            if not combined_results:
                print("Aucun document trouvé")
                return f"Aucun document pertinent trouvé pour '{clean_query}'."
            
            # Extraire un résumé des sources séparément
            source_summary = self._extract_source_summary(combined_results)
            
            # Ajouter des marqueurs spéciaux pour faciliter l'extraction des sources par les agents
            source_section = (
                "### DEBUT_SOURCES_DOCUMENTAIRES ###\n"
                "===== Sources documentaires =====\n"
                f"{source_summary}\n"
                "### FIN_SOURCES_DOCUMENTAIRES ###"
            )
            
            # Formater le contenu principal
            result = self._format_results(combined_results, clean_query, additional_notes)
            
            # Ajouter explicitement la section des sources à la fin
            final_result = f"{result}\n\n{source_section}"
            
            print(f"RetrieverTool: réponse générée ({len(final_result)} caractères)")
            return final_result
            
        except Exception as e:
            error_message = f"Error in retriever tool: {str(e)}"
            print(f"ERREUR RetrieverTool: {error_message}")
            import traceback
            print(f"TRACEBACK COMPLET: {traceback.format_exc()}")
            return f"La recherche a échoué avec l'erreur: {error_message}"
    
    def _preprocess_query(self, query: str) -> str:
        """Nettoie et prépare la requête pour la recherche."""
        # Gestion de la requête vide
        if not query:
            return "information"
            
        # Conversion en minuscules
        query = query.lower().strip()
        
        # Suppression des caractères spéciaux
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Suppression des préfixes communs
        prefixes = DEFINITION_PREFIXES if DEFINITION_PREFIXES else [
            "définition de", "definition of", "concept de", "concept of", 
            "what is", "qu'est-ce que"
        ]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        # Retourner la requête nettoyée
        return query if query else "information"
    
    def _combine_results(self, results1: List[Document], results2: List[Document]) -> List[Document]:
        """Combine les résultats en éliminant les doublons."""
        combined = []
        seen_content = set()
        
        # Fonction pour ajouter un document unique
        def add_if_unique(doc):
            if hasattr(doc, 'page_content') and doc.page_content:
                if doc.page_content not in seen_content:
                    combined.append(doc)
                    seen_content.add(doc.page_content)
        
        # Ajouter les documents des deux listes
        for doc in results1:
            add_if_unique(doc)
        
        for doc in results2:
            add_if_unique(doc)
        
        # Limiter le nombre total de résultats
        global _SEARCH_CONFIG
        return combined[:_SEARCH_CONFIG['max_docs']]
    
    def _try_alternative_queries(self, original_query: str) -> List[Document]:
        """Essaie des variantes de la requête pour trouver plus de résultats."""
        print(f"Essai de requêtes alternatives pour '{original_query}'")
        results = []
        seen_content = set()
        
        # Accès à la variable globale
        global _GLOBAL_VECTORDB
        
        # Générer quelques variantes de requête
        alternative_queries = [
            # Version plus générale 
            " ".join(original_query.split()[:2]) if len(original_query.split()) > 2 else original_query,
            # Ajout de termes utiles
            f"{original_query} concept" if "concept" not in original_query else original_query,
            f"{original_query} definition" if "definition" not in original_query else original_query,
        ]
        
        # Essayer chaque requête alternative
        for alt_query in alternative_queries:
            if alt_query != original_query:
                try:
                    alt_results = _GLOBAL_VECTORDB.similarity_search(alt_query, k=2)
                    for doc in alt_results:
                        if hasattr(doc, 'page_content') and doc.page_content not in seen_content:
                            results.append(doc)
                            seen_content.add(doc.page_content)
                except Exception as e:
                    print(f"Échec de la requête alternative '{alt_query}': {e}")
            
            # Si on a suffisamment de résultats, arrêter
            if len(results) >= 3:
                break
                
        print(f"Requêtes alternatives: {len(results)} documents trouvés")
        return results
    
    def _format_results(self, results: List[Document], query: str, additional_notes: str = None) -> str:
        """Formate les résultats en texte structuré avec sources documentaires visibles."""
        # Créer un résumé détaillé des sources pour l'affichage au début
        summary_header = ["===== RÉSUMÉ DES SOURCES DOCUMENTAIRES ====="]
        
        # Collecter les informations sur les sources pour le résumé
        sources_info = {}
        
        for doc in results:
            # Extraire les métadonnées
            metadata = getattr(doc, 'metadata', {}) or {}
            source = metadata.get('source', 'Unknown source')
            page = metadata.get('page', 'Unknown page')
            
            # Formater le nom de la source pour l'affichage
            source_name = os.path.basename(source) if isinstance(source, str) else "Document inconnu"
            
            # Collecter des informations sur les sources pour le résumé
            if source_name not in sources_info:
                sources_info[source_name] = {
                    'pages': set(),
                }
            sources_info[source_name]['pages'].add(str(page))
        
        # Formater le résumé des sources
        for source_name, info in sources_info.items():
            # Trier les pages numériquement si possible
            page_list = sorted(info['pages'], key=lambda x: int(x) if x.isdigit() else x)
            page_str = ", ".join(page_list)
            summary_header.append(f"📄 {source_name} (pages: {page_str})")
        
        summary_header.append("=" * 40)
        
        # Construire l'en-tête
        header = [
            "Retrieved documents:",
            f"Query: {query}",
        ]
        if additional_notes:
            header.append(f"Context: {additional_notes}")
        header.append(f"Found {len(results)} relevant documents")
        header.append("=" * 40)
        
        # Formater chaque document
        documents = []
        for i, doc in enumerate(results, 1):
            # Extraire les métadonnées
            metadata = getattr(doc, 'metadata', {}) or {}
            source = metadata.get('source', 'Unknown source')
            page = metadata.get('page', 'Unknown page')
            
            # Formater le nom de la source pour l'affichage
            source_name = os.path.basename(source) if isinstance(source, str) else "Document inconnu"
            
            # Obtenir le contenu du document
            content = getattr(doc, 'page_content', 'No content available')
            
            # Construire la section du document
            doc_section = [
                f"===== Document {i} =====",
                f"Source: {source_name}",
                f"Page: {page}",
                "Content:",
                content,
                "=" * 40
            ]
            documents.append("\n".join(doc_section))
        
        # N'incluons PAS la section sources ici car elle est gérée séparément dans forward()
        # pour éviter le doublement des informations
        
        # Assembler le résultat final avec seulement le résumé des sources au début et les documents
        return "\n".join(summary_header + header + documents)
    
    def _extract_source_summary(self, results: List[Document]) -> str:
        """Extrait un résumé des sources documentaires pour affichage client."""
        sources = {}
        for doc in results:
            metadata = getattr(doc, 'metadata', {}) or {}
            source = metadata.get('source', 'Unknown source')
            page = metadata.get('page', 'Unknown page')
            
            # Formater le nom de la source
            source_name = os.path.basename(source) if isinstance(source, str) else "Document inconnu"
            
            # Ajouter la page à la liste des pages pour cette source
            if source_name not in sources:
                sources[source_name] = set()
            sources[source_name].add(str(page))
        
        # Formater le résumé des sources
        summary_lines = []
        for source_name, pages in sources.items():
            # Trier les pages et les joindre en une chaîne
            page_list = sorted(pages, key=lambda x: int(x) if x.isdigit() else x)
            page_str = ", ".join(page_list)
            summary_lines.append(f"- {source_name} (pages: {page_str})")
        
        return "\n".join(summary_lines) if summary_lines else "Aucune source identifiée"

# Fonctions globales d'accès à la base de données vectorielle
def get_vectordb():
    """Retourne la base de données vectorielle globale."""
    global _GLOBAL_VECTORDB
    return _GLOBAL_VECTORDB

def set_vectordb(vectordb):
    """Définit la base de données vectorielle globale."""
    global _GLOBAL_VECTORDB
    _GLOBAL_VECTORDB = vectordb
    return True