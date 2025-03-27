import os
import re
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

class PDFProcessor:
    def __init__(
        self, 
        chroma_path: str = "./chroma_db",
        collection_name: str = "pdf_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chroma_path: Path to store the ChromaDB
            collection_name: Name of the collection in ChromaDB
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            model_name: HuggingFace model for embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
    def extract_text_and_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text and metadata
        """
        # Use PyMuPDF to extract text and metadata
        doc = fitz.open(pdf_path)
        
        # Extract text
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Extract metadata
        metadata = {
            "filename": os.path.basename(pdf_path),
            "path": pdf_path,
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }
        
        # Try to extract additional data from text for missing metadata
        if not metadata["title"]:
            title_match = re.search(r'^(.+)\n', text)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
        
        # Extract potential keywords if not in metadata
        if not metadata["keywords"]:
            # Simple keyword extraction based on frequency
            words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                if word not in ["this", "that", "with", "from", "have", "were", "they", "their"]:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 5 most frequent words as keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            metadata["extracted_keywords"] = ", ".join([k for k, v in keywords])
        
        doc.close()
        return {"text": text, "metadata": metadata}
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Process a PDF file: extract text, create chunks, and add to ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document IDs added to the database
        """
        # Extract text and metadata
        data = self.extract_text_and_metadata(pdf_path)
        text = data["text"]
        metadata = data["metadata"]
        
        # Create chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add chunks to ChromaDB
        doc_ids = []
        for i, chunk in enumerate(chunks):
            # Create unique ID for the chunk
            doc_id = f"{os.path.basename(pdf_path)}_{i}"
            doc_ids.append(doc_id)
            
            # Create metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            # Get embedding for the chunk
            embedding = self.embeddings.embed_query(chunk)
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
                documents=[chunk]
            )
        
        return doc_ids
    
    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            Dictionary mapping filenames to their document IDs
        """
        results = {}
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                try:
                    doc_ids = self.process_pdf(file_path)
                    results[filename] = doc_ids
                    print(f"Processed {filename}: {len(doc_ids)} chunks added")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return results
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the database for relevant chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Get embedding for the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "id": results['ids'][0][i]
                })
        
        return formatted_results


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Process a single PDF
    # doc_ids = processor.process_pdf("path/to/your/document.pdf")
    # print(f"Added {len(doc_ids)} chunks to the database")
    
    # Process a directory of PDFs
    # results = processor.process_directory("path/to/your/pdf/directory")
    # print(f"Processed {len(results)} PDF files")
    
    # Search example
    # results = processor.search("your search query")
    # for i, result in enumerate(results):
    #     print(f"Result {i+1}:")
    #     print(f"Text: {result['text'][:100]}...")
    #     print(f"Metadata: {result['metadata']}")
    #     print("-" * 50)
