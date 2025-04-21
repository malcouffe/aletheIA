import os
import streamlit as st # For potential feedback/errors
from smolagents import (
    CodeAgent, 
    ToolCallingAgent, 
    DuckDuckGoSearchTool, 
    VisitWebpageTool, 
    OpenAIServerModel
)
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from sentence_transformers import SentenceTransformer
from managed_agent.retriever_tool import RetrieverTool

# Define embedding model name centrally
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def initialize_search_agent(model: OpenAIServerModel) -> ToolCallingAgent:
    """Initializes the web search agent."""
    return ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        description="Effectue des recherches sur le web en utilisant DuckDuckGo et visite des pages web."
    )

def initialize_data_analyst_agent(model: OpenAIServerModel) -> CodeAgent:
    """Initializes the data analyst agent."""
    authorized_imports = [
        "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
        "seaborn", "io", "base64", "tempfile", "os"
    ]
    return CodeAgent(
        tools=[], # Specific tools like csv_analyzer are passed via additional_args
        model=model,
        additional_authorized_imports=authorized_imports,
        name="data_analyst",
        description="Analyse les fichiers CSV et génère des visualisations à partir des données."
    )

def initialize_rag_agent(model: OpenAIServerModel, db_path: str) -> CodeAgent | None:
    """Initializes the RAG agent if the vector database exists."""
    try:
        print(f"RAG Agent: Attempting to load vector store from: {db_path}")
        # Ensure the directory exists before trying to load
        if not os.path.exists(db_path):
             raise FileNotFoundError(f"Vector DB path not found: {db_path}")

        # 1. Initialize the embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 2. Initialize the Chroma vector store
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
            # collection_name="pdf_collection" # Assuming default
        )

        # 3. Initialize the RetrieverTool
        retriever_tool = RetrieverTool(vectordb=vector_store)

        # 4. Initialize the RAG Agent
        rag_agent = CodeAgent(
            tools=[retriever_tool],
            model=model,
            name="rag_agent",
            max_steps=4,
            verbosity_level=2,
            description="Effectue une recherche RAG sur le document PDF indexé pour répondre à la requête."
        )
        print("RAG Agent initialized successfully.")
        return rag_agent

    except Exception as rag_init_error:
        st.error(f"Erreur lors de l'initialisation du RAG Agent : {rag_init_error}")
        print(f"RAG Agent initialization failed: {rag_init_error}")
        return None

def route_request(query: str, csv_args: dict | None, search_agent: ToolCallingAgent, data_analyst: CodeAgent, rag_agent: CodeAgent | None = None, pdf_context: dict | None = None):
    """
    Fonction de routage qui délègue la requête à l'agent spécialisé.
    Priorité : RAG si applicable > Data Analysis (CSV) > Recherche Web.
    """
    # --- RAG Agent Check ---
    # Prioritize RAG if it's initialized and the query is likely related to the PDF
    # (Simple check: RAG exists and query is not empty. Could be refined.)
    if rag_agent and query:
        print(f"Routing to RAG agent.")
        rag_query = query
        # Construct context string if pdf_context is provided (optional)
        if pdf_context and pdf_context.get('summary'):
             context_info = f"Context from PDF ({pdf_context.get('classification', 'N/A')}) Summary: {pdf_context.get('summary', '')[:200]}..."
             rag_query += f"\n\n{context_info}"
             print(f"  with context: {context_info}")
        # Pass the query (potentially with context) to the RAG agent
        # The RetrieverTool within the RAG agent handles the actual retrieval
        return rag_agent.run(rag_query)

    # --- Data Analyst Check ---
    elif csv_args is not None:
        print("Routing to Data Analyst agent.")
        additional_notes = csv_args.get('additional_notes', '').strip()

        expertise_message = (
            "Vous êtes un expert en data-analysis. "
            "Votre tâche est d'analyser le fichier CSV fourni afin de répondre à la question posée. "
        )

        prompt = (
            f"{expertise_message}\n"
            f"Analyse du fichier CSV: {query}\n\n"
            f"Notes additionnelles et contexte:\n{additional_notes}"
        )

        # Prepare args for the csv_analyzer tool (assuming it's implicitly available or passed)
        csv_analyzer_args = {
            "source_file": csv_args["source_file"],
            "separator": csv_args["separator"],
            "figures_dir": csv_args["figures_dir"],
            "chunk_size": csv_args["chunk_size"]
        }

        # Run the data analyst agent with the prepared prompt and tool arguments
        return data_analyst.run(prompt, additional_args={"csv_analyzer": csv_analyzer_args})

    # --- Default to Search Agent ---
    else:
        print("Routing to Search agent.")
        # Delegate to the search agent for general queries
        return search_agent.run(query) 