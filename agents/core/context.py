"""
Gestion du Contexte pour les Systèmes Multi-Agents
Gère la préparation et le passage du contexte entre les agents suivant les meilleures pratiques smolagents.
"""

import logging
from typing import List, Dict, Any, Optional

# Configuration du logging pour le débogage de la gestion du contexte - désactive les logs HTTP/API
logging.basicConfig(level=logging.WARNING)  # Définit le niveau global à WARNING
logger = logging.getLogger(__name__)

# Désactive les logs verbeux HTTP/API
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class ContextManager:
    """Gère les données de contexte pour les conversations multi-agents."""
    
    def __init__(self):
        self.pdf_context: Optional[Dict] = None
        self.csv_context: Optional[Dict] = None
    
    def set_pdf_context(self, available_pdfs: List[Dict]):
        """Définit le contexte PDF à partir des fichiers PDF disponibles."""
        if available_pdfs:
            self.pdf_context = {
                "available_files": available_pdfs,
                "count": len(available_pdfs),
                "classifications": list(set(
                    pdf.get('classification', 'Général') 
                    for pdf in available_pdfs
                ))
            }
        else:
            print("⚠️ Aucun fichier PDF fourni")
    
    def set_csv_context(self, available_csvs: List[Dict]):
        """Définit le contexte CSV à partir des fichiers CSV disponibles."""
        if available_csvs:
            self.csv_context = {
                "available_files": available_csvs,
                "count": len(available_csvs),
                "total_columns": sum(
                    len(csv.get('csv_args', {}).get('columns', [])) 
                    for csv in available_csvs
                )
            }
        else:
            print("⚠️ Aucun fichier CSV fourni")
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Récupère le dictionnaire de contexte complet pour les arguments additionnels de l'agent."""
        context = {}
        
        if self.pdf_context:
            context["pdf_context"] = self.pdf_context
        
        if self.csv_context:
            context["csv_context"] = self.csv_context
        
        return context
    
    def get_context_summary(self) -> List[str]:
        """Récupère un résumé lisible du contexte pour les descriptions de tâches."""
        context_hints = []
        
        if self.pdf_context:
            pdf_count = self.pdf_context["count"]
            classifications = self.pdf_context["classifications"]
            hint = f"Documents PDF : {pdf_count} fichiers disponibles ({', '.join(classifications)} sujets)"
            context_hints.append(hint)
        
        if self.csv_context:
            csv_count = self.csv_context["count"]
            hint = f"Jeux de données CSV : {csv_count} fichiers disponibles pour l'analyse"
            context_hints.append(hint)
        
        if not context_hints:
            hint = "Capacités de recherche web"
            context_hints.append(hint)
        
        return context_hints


def prepare_manager_context(additional_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prépare les données de contexte pour l'agent gestionnaire en utilisant les arguments additionnels.
    
    Args:
        additional_args: Dictionnaire contenant les informations de contexte
    
    Returns:
        Dictionnaire de contexte pour les arguments additionnels
    """
    # Extraction du contexte des arguments additionnels
    pdf_context = additional_args.get('pdf_context', {})
    csv_context = additional_args.get('csv_context', {})
    language = additional_args.get('language', 'fr')  # Langue par défaut
    
    available_pdfs = pdf_context.get('available_files', [])
    available_csvs = csv_context.get('available_files', [])
    
    manager = ContextManager()
    
    if available_pdfs:
        manager.set_pdf_context(available_pdfs)
    
    if available_csvs:
        manager.set_csv_context(available_csvs)
    
    context = manager.get_context_dict()
    context['language'] = language  # Ajout de la langue au contexte
    
    return context


def build_simple_manager_task(user_query: str, context: Dict[str, Any]) -> str:
    """
    Construit une description de tâche simple pour l'agent gestionnaire.
    
    Args:
        user_query: La requête/demande de l'utilisateur (déjà enrichie par l'agent contextuel)
        context: Dictionnaire contenant les informations de contexte
    
    Returns:
        Description de tâche formatée optimisée pour la délégation smolagents
    """
    manager = ContextManager()
    
    # Extraction du contexte des arguments additionnels
    pdf_context = context.get('pdf_context', {})
    csv_context = context.get('csv_context', {})
    language = context.get('language', 'fr')  # Langue par défaut
    
    available_pdfs = pdf_context.get('available_files', [])
    available_csvs = csv_context.get('available_files', [])
    
    if available_pdfs:
        manager.set_pdf_context(available_pdfs)
    
    if available_csvs:
        manager.set_csv_context(available_csvs)
    
    context_hints = manager.get_context_summary()
    
    # Format de tâche amélioré pour une meilleure délégation smolagents
    task = f"""Requête Enrichie : "{user_query}"

Ressources Disponibles :
{chr(10).join(f"- {hint}" for hint in context_hints)}

Langue Cible : {language}

Instructions pour le Gestionnaire :
- Analyser la requête enrichie et les ressources disponibles
- Déléguer IMMÉDIATEMENT à l'agent spécialiste approprié
- Utiliser la journalisation de débogage pour montrer la décision de délégation
- Passer la requête enrichie complète au spécialiste
- S'assurer que la réponse est dans la langue cible spécifiée"""
    
    return task


 