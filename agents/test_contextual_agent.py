"""
Script de test pour l'agent contextuel et sa capacité de reformulation.
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from smolagents import OpenAIServerModel
from agents.agent_manager_multiagent import ContextualAgent

def test_contextual_agent():
    # Vérification de la clé API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement (OPENAI_API_KEY)")

    # Initialisation du modèle
    print("🚀 Initialisation du modèle...")
    model = OpenAIServerModel(
        model_id="gpt-3.5-turbo",
        api_base="https://api.openai.com/v1",
        api_key=api_key
    )
    
    # Initialisation de l'agent contextuel
    print("🏗️ Initialisation de l'agent contextuel...")
    contextual_agent = ContextualAgent(model)
    
    # Test 1: Requête simple sans contexte
    print("\n📝 Test 1: Requête simple sans contexte")
    query1 = "Peux-tu me donner des graphiques sur ce dataset ?"
    print(f"Requête originale: {query1}")
    result1 = contextual_agent.process_query(query1)
    print(f"Résultat reformulé: {result1}")
    
    # Test 2: Requête avec contexte de fichier CSV
    print("\n📝 Test 2: Requête avec contexte de fichier CSV")
    contextual_agent.update_file_context("csv", "bank_transaction_light.csv")
    query2 = "Analyse ce fichier"
    print(f"Requête originale: {query2}")
    result2 = contextual_agent.process_query(query2)
    print(f"Résultat reformulé: {result2}")
    
    # Test 3: Requête avec contexte de fichier PDF
    print("\n📝 Test 3: Requête avec contexte de fichier PDF")
    contextual_agent.update_file_context("pdf", "rapport_financier.pdf")
    query3 = "Trouve les informations sur les contrôles internes"
    print(f"Requête originale: {query3}")
    result3 = contextual_agent.process_query(query3)
    print(f"Résultat reformulé: {result3}")
    
    # Test 4: Requête avec historique d'interactions
    print("\n📝 Test 4: Requête avec historique d'interactions")
    contextual_agent.process_query("Montre-moi les transactions du mois dernier")
    contextual_agent.process_query("Fais un graphique des dépenses par catégorie")
    query4 = "Fais la même chose pour ce mois-ci"
    print(f"Requête originale: {query4}")
    result4 = contextual_agent.process_query(query4)
    print(f"Résultat reformulé: {result4}")
    
    # Test 5: Requête avec contexte mixte (CSV + PDF)
    print("\n📝 Test 5: Requête avec contexte mixte")
    contextual_agent.update_file_context("csv", "donnees_2024.csv")
    contextual_agent.update_file_context("pdf", "analyse_2024.pdf")
    query5 = "Compare les données avec le rapport"
    print(f"Requête originale: {query5}")
    result5 = contextual_agent.process_query(query5)
    print(f"Résultat reformulé: {result5}")
    
    # Affichage du contexte final
    print("\n📊 Contexte final de l'agent:")
    print(f"Fichiers récents: {contextual_agent.context['recent_files']}")
    print(f"Dernier CSV: {contextual_agent.context['dernier_csv']}")
    print(f"Dernier PDF: {contextual_agent.context['dernier_pdf']}")
    print(f"Nombre d'interactions: {len(contextual_agent.context['interaction_history'])}")

if __name__ == "__main__":
    print("🧪 Démarrage des tests de l'agent contextuel...")
    test_contextual_agent()
    print("\n✅ Tests terminés !") 