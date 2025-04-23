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
from langchain_community.embeddings import HuggingFaceEmbeddings 
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
        if not os.path.exists(db_path):
             raise FileNotFoundError(f"Vector DB path not found: {db_path}")

        # 1. Initialize the embedding model using the LangChain wrapper
        # Use cache_folder to potentially speed up loading on subsequent runs
        model_kwargs = {'device': 'cpu'} # Or 'cuda' if GPU available and configured
        encode_kwargs = {'normalize_embeddings': False}
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # cache_folder=os.path.join(os.getcwd(), ".cache") # Optional: Specify a cache folder
        )

        # 2. Initialize the Chroma vector store with the wrapped embedding function
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function
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

def initialize_manager_agent(model: OpenAIServerModel) -> CodeAgent:
    """Initializes the Manager CodeAgent."""
    return CodeAgent(
        tools=[], # Manager doesn't directly use tools, it delegates
        model=model,
        name="manager_agent",
        description=(
            "Acts as a router. Analyzes the user query and context (PDF indexed, CSV loaded) "
            "and decides which specialized agent (search, data_analyst, rag_agent) to call. "
            "The specialized agents are provided in additional_args."
        )
    ) 