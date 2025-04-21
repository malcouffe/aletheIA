from smolagents import Tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "additional_notes": {
            "type": "string",
            "description": "Optional additional notes or context to refine the search query.",
            "optional": True,
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str, additional_notes: str | None = None) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        # Combine query and notes if provided
        effective_query = query
        if additional_notes and isinstance(additional_notes, str) and additional_notes.strip():
            effective_query = f"{query}\n\nAdditional Context/Notes:\n{additional_notes.strip()}"
            print(f"RetrieverTool using combined query: {effective_query[:200]}...") # Log combined query
        else:
            print(f"RetrieverTool using query: {query[:200]}...")

        docs = self.vectordb.similarity_search(
            effective_query, # Use the potentially combined query
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )