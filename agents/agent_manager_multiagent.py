"""
Interface de Gestion Multi-Agents Simplifiée
Routage simple basé sur le type de fichier détecté dans la requête.
Avec support des step callbacks selon les bonnes pratiques smolagents.
"""

from smolagents import OpenAIServerModel, ToolCallingAgent, CodeAgent
from typing import Optional, Dict, Any, List, Callable
import re

from agents.factories.agent_factory_multiagent import create_multiagent_system
from agents.config.agent_config import USER_QUERY_PREPROMPTS, PREPROMPT_CONFIG


class SimplifiedMultiAgentManager:
    """
    Système multi-agents simplifié avec routage direct basé sur le type de fichier.
    - PDF dans la requête -> RAG agent
    - CSV dans la requête -> Data analyst agent  
    - Reste -> Search agent
    
    Supporte les step callbacks selon les bonnes pratiques smolagents.
    """
    
    def __init__(self, model: OpenAIServerModel):
        """
        Initialize the simplified multi-agent manager.
        
        Args:
            model: OpenAI model to use for all agents
        """
        self.model = model
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        self._initialized = False
    
    def _inject_preprompt(self, user_query: str, agent_type: str) -> str:
        """
        Injecte le pré-prompt approprié avec la requête utilisateur.
        
        Args:
            user_query: La requête originale de l'utilisateur
            agent_type: Type d'agent ('rag_agent', 'data_analyst', 'search_agent')
            
        Returns:
            Requête avec pré-prompt injecté
        """
        if not PREPROMPT_CONFIG.get("enabled", False):
            return user_query
        
        # Sélectionner le bon pré-prompt
        preprompt = USER_QUERY_PREPROMPTS.get(agent_type, USER_QUERY_PREPROMPTS["general"])
        
        # Construire la requête avec pré-prompt
        separator = PREPROMPT_CONFIG.get("separator", "\n---\n")
        
        if PREPROMPT_CONFIG.get("position", "before") == "before":
            enhanced_query = f"{preprompt}{separator}REQUÊTE UTILISATEUR: {user_query}"
        else:
            enhanced_query = f"REQUÊTE UTILISATEUR: {user_query}{separator}{preprompt}"
        
        print(f"🎯 Injection du pré-prompt pour {agent_type}")
        return enhanced_query
    
    def _detect_file_type(self, query: str) -> str:
        """
        Détecte le type de fichier dans la requête pour router vers le bon agent.
        
        Args:
            query: La requête de l'utilisateur
            
        Returns:
            'pdf' si PDF détecté, 'csv' si CSV détecté, 'web' sinon
        """
        query_lower = query.lower()
        
        # Détection PDF (plus flexible et étendue)
        pdf_patterns = [
            r'\bpdf\b',
            r'\.pdf\b',
            r'\bdocument\b',
            r'\brapport\b',
            r'\bfichier pdf\b',
            # Nouveaux patterns pour mieux détecter les requêtes sur documents
            r'\bque dit\b',
            r'\bselon le document\b',
            r'\bdans le fichier\b',
            r'\bextrait\b',
            r'\bpage\b',
            r'\bpassage\b',
            r'\bcitation\b',
            r'\bréférences?\b',
            r'\bméthodologie\b',
            r'\bannexe\b',
            r'\bsection\b',
            r'\banalyse documentaire\b'
        ]
        
        # Détection CSV (plus flexible)  
        csv_patterns = [
            r'\bcsv\b',
            r'\.csv\b',
            r'\bdonnées\b',
            r'\bdataset\b',
            r'\btableau\b',
            r'\banalyse de données\b',
            r'\bfichier csv\b',
            r'\bgraphique\b',
            r'\bvisualis\b',
            r'\bstatistiques?\b'
        ]
        
        # Vérifier PDF en premier
        for pattern in pdf_patterns:
            if re.search(pattern, query_lower):
                print(f"🎯 Agent Router: Detected PDF pattern '{pattern}' in query")
                return 'pdf'
        
        # Vérifier CSV ensuite
        for pattern in csv_patterns:
            if re.search(pattern, query_lower):
                print(f"🎯 Agent Router: Detected CSV pattern '{pattern}' in query")
                return 'csv'
        
        # Par défaut: recherche web
        print(f"🎯 Agent Router: No specific file type detected, defaulting to web search")
        return 'web'
    
    def process_query(self, prompt: str, model, available_pdfs_context: str, 
                     available_csvs_context: str, step_callbacks: List[Callable] = None) -> str:
        """
        Process query with smolagents best practices support (step callbacks).
        
        Args:
            prompt: User query
            model: Model to use (unused, uses self.model)
            available_pdfs_context: PDF context
            available_csvs_context: CSV context  
            step_callbacks: List of callback functions to execute after each step
            
        Returns:
            Response from the appropriate agent
        """
        if not self._initialized:
            raise RuntimeError("Multi-agent system not initialized. Call initialize() first.")
        
        # Détection du type de fichier
        file_type = self._detect_file_type(prompt)
        
        print(f"🎯 Processing query: {prompt}")
        print(f"📁 Detected file type: {file_type}")
        
        try:
            # Routage vers l'agent approprié avec injection de pré-prompt
            if file_type == 'pdf':
                print("📄 Routing to RAG agent (PDF processing)")
                agent = self.document_agent
                agent_type = 'rag_agent'
            elif file_type == 'csv':
                print("📊 Routing to Data Analyst agent (CSV analysis)")
                agent = self.data_analyst_agent
                agent_type = 'data_analyst'
            else:
                print("🔍 Routing to Search agent (Web research)")
                agent = self.search_agent
                agent_type = 'search_agent'
            
            # Injecter le pré-prompt avec la requête
            enhanced_prompt = self._inject_preprompt(prompt, agent_type)
            
            # Configure step callbacks if provided (smolagents best practice)
            if step_callbacks and hasattr(agent, 'step_callbacks'):
                print(f"🔄 SMOLAGENTS: Configuring {len(step_callbacks)} step callbacks")
                agent.step_callbacks = step_callbacks
            
            # Execute the task
            result = agent.run(enhanced_prompt)
            
            print("✅ Query processed successfully!")
            return str(result)
            
        except Exception as e:
            print(f"❌ Error during query processing: {str(e)}")
            raise

    def run_task(self, user_query: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Exécute une tâche en routant directement vers l'agent approprié.
        
        Args:
            user_query: La requête de l'utilisateur
            additional_args: Arguments additionnels (non utilisés dans la version simplifiée)
            
        Returns:
            Le résultat de la tâche
        """
        if not self._initialized:
            raise RuntimeError("Multi-agent system not initialized. Call initialize() first.")
        
        # Détection du type de fichier
        file_type = self._detect_file_type(user_query)
        
        print(f"🎯 Processing task: {user_query}")
        print(f"📁 Detected file type: {file_type}")
        
        try:
            # Routage direct vers l'agent approprié avec injection de pré-prompt
            if file_type == 'pdf':
                print("📄 Routing to RAG agent (PDF processing)")
                enhanced_query = self._inject_preprompt(user_query, 'rag_agent')
                result = self.document_agent.run(enhanced_query)
            elif file_type == 'csv':
                print("📊 Routing to Data Analyst agent (CSV analysis)")
                enhanced_query = self._inject_preprompt(user_query, 'data_analyst')
                result = self.data_analyst_agent.run(enhanced_query)
            else:
                print("🔍 Routing to Search agent (Web research)")
                enhanced_query = self._inject_preprompt(user_query, 'search_agent')
                result = self.search_agent.run(enhanced_query)
            
            print("✅ Task completed successfully!")
            return str(result)
            
        except Exception as e:
            print(f"❌ Error during task execution: {str(e)}")
            raise

    def initialize(self):
        """Initialize the simplified multi-agent system."""
        if self._initialized:
            print("⚠️ Multi-agent system already initialized")
            return
        
        print("🚀 Initializing simplified multi-agent system...")
        
        # Create specialized agents (no manager needed)
        (
            _,  # Skip manager agent (None)
            self.data_analyst_agent,
            self.document_agent,
            self.search_agent
        ) = create_multiagent_system(self.model)
        
        self._initialized = True
        print("✅ Simplified multi-agent system initialized!")
        print("   Routing Logic:")
        print("   ├── PDF keywords → RAG Agent (document processing)")
        print("   ├── CSV keywords → Data Analyst Agent (data analysis)")  
        print("   └── Everything else → Search Agent (web research)")

    def reset(self):
        """Reset the multi-agent system."""
        self._initialized = False
        self.data_analyst_agent = None
        self.document_agent = None
        self.search_agent = None
        print("🔄 Multi-agent system reset")


def initialize_multiagent_system(model: OpenAIServerModel) -> SimplifiedMultiAgentManager:
    """
    Initialize a new simplified multi-agent system.
    
    Args:
        model: OpenAI model to use for all agents
        
    Returns:
        Initialized SimplifiedMultiAgentManager instance
    """
    manager = SimplifiedMultiAgentManager(model)
    manager.initialize()
    return manager


# Backward compatibility
def initialize_simplified_agent_system(model: OpenAIServerModel) -> SimplifiedMultiAgentManager:
    """
    Backward compatibility alias.
    
    Args:
        model: OpenAI model to use
        
    Returns:
        Initialized SimplifiedMultiAgentManager
    """
    return initialize_multiagent_system(model)


# Legacy compatibility (old class name)
MultiAgentManager = SimplifiedMultiAgentManager 