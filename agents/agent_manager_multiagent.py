"""
Multi-Agent Manager Interface
Implements smolagents best practices with specialized agents and minimal manager.
"""

from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from typing import Optional, List, Dict, Any
import openai

from agents.factories.agent_factory_multiagent import create_multiagent_system
from agents.core.context import prepare_manager_context, build_simple_manager_task
from agents.core.embedding import get_embedding_function


@tool
def reformulate_query(query: str, context_str: str) -> str:
    """
    Reformule une requête en tenant compte du contexte.
    
    Args:
        query: La requête originale à reformuler
        context_str: Le contexte sous forme de chaîne de caractères
        
    Returns:
        La requête reformulée
    """
    return query  # L'agent smolagents s'occupera de la reformulation


class ContextualAgent:
    """
    Agent spécialisé dans la reformulation et l'enrichissement du contexte.
    Suit un processus d'action en chaîne bien défini pour chaque requête.
    """
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.context = {
            "recent_files": [],
            "dernier_csv": None,
            "dernier_pdf": None,
            "interaction_history": []
        }
        self.max_history = 20
        
        # Initialisation de l'agent smolagents pour la reformulation
        self.reformulation_agent = ToolCallingAgent(
            model=self.model,
            tools=[reformulate_query],
            name="reformulation_agent",
            description="Agent spécialisé dans la reformulation de requêtes en tenant compte du contexte.",
            max_steps=1,
            verbosity_level=0
        )

    def process_query(self, query: str) -> str:
        """
        Processus en chaîne pour traiter une requête :
        1. Analyse du contexte actuel
        2. Enrichissement avec l'historique
        3. Reformulation de la requête
        4. Mise à jour du contexte
        
        Args:
            query: Requête originale de l'utilisateur
            
        Returns:
            Requête enrichie et reformulée
        """
        print("🔄 Contextual Agent: Début du traitement de la requête")
        print(f"📝 Requête originale: {query}")
        
        # 1. Analyse du contexte actuel
        current_context = self._analyze_current_context()
        print("📊 Contexte actuel analysé")
        
        # 2. Enrichissement avec l'historique
        enriched_context = self._enrich_with_history(current_context)
        print("📚 Contexte enrichi avec l'historique")
        
        # 3. Reformulation de la requête
        print("🔄 Début de la reformulation de la requête...")
        reformulated_query = self._reformulate_query(query, enriched_context)
        print(f"✅ Requête reformulée: {reformulated_query}")
        
        # 4. Mise à jour du contexte
        self._update_context(query, reformulated_query)
        print("📝 Contexte mis à jour avec la nouvelle interaction")
        
        return reformulated_query

    def _analyze_current_context(self) -> Dict[str, Any]:
        """
        Analyse le contexte actuel pour identifier les éléments pertinents.
        """
        return {
            "recent_files": self.context["recent_files"][-5:],
            "dernier_csv": self.context["dernier_csv"],
            "dernier_pdf": self.context["dernier_pdf"],
            "last_interactions": self.context["interaction_history"][-3:]
        }

    def _enrich_with_history(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrichit le contexte avec l'historique des interactions.
        """
        enriched = current_context.copy()
        enriched["full_history"] = self.context["interaction_history"][-self.max_history:]
        return enriched

    def _reformulate_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Reformule la requête en tenant compte du contexte enrichi.
        """
        print("🔄 Début de la reformulation avec le modèle OpenAI...")
        
        try:
            print("📡 Appel à l'API OpenAI pour la reformulation...")
            
            # Construction d'un prompt plus détaillé pour la reformulation
            context_str = f"""
            Contexte actuel :
            - Fichiers récemment utilisés : {[f['name'] for f in context['recent_files']]}
            - Dernier fichier CSV utilisé : {context['dernier_csv']}
            - Dernier fichier PDF utilisé : {context['dernier_pdf']}
            - Historique des interactions : {[h['content'] for h in context['last_interactions']]}
            
            Instructions pour la reformulation :
            1. La requête reformulée DOIT faire référence explicitement aux fichiers concernés (CSV ou PDF)
            2. Si la requête fait référence à "ce fichier" ou "le fichier", spécifier le nom exact du fichier
            3. Si la requête fait référence à "le rapport" ou "les données", préciser s'il s'agit du CSV ou du PDF
            4. La reformulation doit être claire et précise, sans phrases d'introduction ou de conclusion
            5. Conserver l'intention originale de la requête tout en la rendant plus spécifique
            """
            
            response = self.reformulation_agent.run(
                f"Reformule cette requête en la rendant plus précise et en faisant référence explicite aux fichiers : {query}",
                additional_args={
                    "query": query,
                    "context_str": context_str
                }
            )
            
            reformulated_query = response.strip()
            print(f"✅ Réponse du modèle reçue: {reformulated_query}")
            return reformulated_query if reformulated_query else query
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la reformulation : {e}")
            print(f"⚠️ Type d'erreur : {type(e)}")
            print(f"⚠️ Détails de l'erreur : {str(e)}")
            return query

    def _update_context(self, original_query: str, reformulated_query: str) -> None:
        """
        Met à jour le contexte avec la nouvelle interaction.
        """
        # Ajout de l'interaction à l'historique
        self.context["interaction_history"].append({
            "type": "user",
            "content": original_query
        })
        self.context["interaction_history"].append({
            "type": "assistant",
            "content": reformulated_query
        })
        
        # Limiter la taille de l'historique
        if len(self.context["interaction_history"]) > self.max_history * 2:
            self.context["interaction_history"] = self.context["interaction_history"][-self.max_history * 2:]

    def update_file_context(self, file_type: str, file_name: str) -> None:
        """
        Met à jour le contexte avec les informations sur le fichier utilisé.
        """
        self.context[f"dernier_{file_type}"] = file_name
        self.context["recent_files"].append({
            "type": file_type,
            "name": file_name,
            "timestamp": "now"  # On pourrait ajouter un vrai timestamp si nécessaire
        })
        # Garder seulement les 5 derniers fichiers
        self.context["recent_files"] = self.context["recent_files"][-5:]

    def get_context(self) -> Dict[str, Any]:
        """
        Récupère le contexte complet.
        """
        return self.context

    def reset_context(self) -> None:
        """
        Réinitialise le contexte.
        """
        self.context = {
            "recent_files": [],
            "dernier_csv": None,
            "dernier_pdf": None,
            "interaction_history": []
        }


class MultiAgentManager:
    """
    Multi-agent system following smolagents best practices.
    Manager delegates to specialized agents rather than doing work directly.
    Now includes a dedicated contextual agent for query reformulation and context management.
    """
    
    def __init__(self, model: OpenAIServerModel):
        """
        Initialize the multi-agent manager with a model.
        
        Args:
            model: OpenAI model to use for all agents
        """
        self.model = model
        self.manager_agent = None
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        self.contextual_agent = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the multi-agent system."""
        if self._initialized:
            print("⚠️ Multi-agent system already initialized")
            return
        
        print("🚀 Initializing multi-agent system...")
        
        # Create the multi-agent system
        (
            self.manager_agent,
            self.data_analyst_agent,
            self.document_agent,
            self.search_agent
        ) = create_multiagent_system(self.model)
        
        # Initialize contextual agent
        self.contextual_agent = ContextualAgent(self.model)
        
        self._initialized = True
        print("✅ Multi-agent system initialized successfully!")
        print("   Architecture:")
        print("   ├── Manager Agent (minimal tools, coordinates everything)")
        print("   ├── Contextual Agent (query reformulation & context management)")
        print("   ├── Data Analyst Agent (CSV analysis & visualizations)")
        print("   ├── Document Agent (document processing)")
        print("   └── Search Agent (web research & information gathering)")

    def _format_final_response(self, result: Any) -> str:
        """
        Format the final response as a string, handling different response types.
        
        Args:
            result: The result from the agent
            
        Returns:
            Formatted string response
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Format dictionary response
            formatted_parts = []
            
            # Add task outcome if available
            if 'task_outcome_detailed' in result:
                formatted_parts.append(result['task_outcome_detailed'])
            elif 'task_outcome_short' in result:
                formatted_parts.append(result['task_outcome_short'])
            
            # Add additional context if available
            if 'additional_context' in result:
                formatted_parts.append("\nContexte supplémentaire:")
                formatted_parts.append(result['additional_context'])
            
            return "\n".join(formatted_parts)
        elif hasattr(result, 'observations'):
            return str(result.observations)
        else:
            return str(result)

    def run_task(self, user_query: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a task through the multi-agent system.
        
        Args:
            user_query: The user's query
            additional_args: Additional arguments for context
            
        Returns:
            The result of the task
        """
        if not self._initialized:
            raise RuntimeError("Multi-agent system not initialized. Call initialize() first.")
        
        print(f"🎯 Processing task: {user_query}...")
        
        # 1. Process through contextual agent
        reformulated_query = self.contextual_agent.process_query(user_query)
        print(f"🔄 Reformulated query: {reformulated_query}")
        
        # 2. Prepare context for manager
        manager_context = prepare_manager_context(additional_args or {})
        print("🏗️ Context Prep: Preparing manager context")
        
        # 3. Build task description with context
        task_description = build_simple_manager_task(reformulated_query, manager_context)
        print("📝 Task Builder: Building manager task description")
        
        # 4. Run task through manager agent
        try:
            result = self.manager_agent.run(task_description)
            print("✅ Task completed successfully!")
            
            # Format the final response
            formatted_result = self._format_final_response(result)
            return formatted_result
            
        except Exception as e:
            print(f"❌ Error during task execution: {str(e)}")
            raise

    def reset(self):
        """Reset the multi-agent system."""
        self._initialized = False
        self.manager_agent = None
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        self.contextual_agent = None
        print("🔄 Multi-agent system reset")

    def get_context(self) -> Dict[str, Any]:
        """Get the current context from the contextual agent."""
        if not self._initialized or not self.contextual_agent:
            return {}
        return self.contextual_agent.get_context()


def initialize_multiagent_system(model: OpenAIServerModel) -> MultiAgentManager:
    """
    Initialize a new multi-agent system.
    
    Args:
        model: OpenAI model to use for all agents
        
    Returns:
        Initialized MultiAgentManager instance
    """
    manager = MultiAgentManager(model)
    manager.initialize()
    return manager


# Backward compatibility - alias for easy migration
def initialize_simplified_agent_system(model: OpenAIServerModel) -> MultiAgentManager:
    """
    Backward compatibility alias for existing code.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized MultiAgentManager (new architecture)
    """
    print("📢 Note: Migrating to new multi-agent architecture (smolagents best practices)")
    return initialize_multiagent_system(model) 