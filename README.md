# PDF Indexer

Cette application permet d'indexer des fichiers PDF en utilisant des embeddings et de stocker les données dans une base de données ChromaDB pour une recherche sémantique efficace.

## Installation

```bash
pip install -r requirements.txt
```

## Fonctionnalités

- Import de documents PDF
- Découpage des documents en chunks
- Extraction des métadonnées (auteur, date, mots-clés, etc.)
- Embedding des chunks avec HuggingFace
- Stockage dans une base de données ChromaDB

## Utilisation

### Indexer un seul fichier PDF

```python
from pipeline_indexation.pdf import PDFProcessor

processor = PDFProcessor()
doc_ids = processor.process_pdf("path/to/your/document.pdf")
print(f"Added {len(doc_ids)} chunks to the database")
```

### Indexer un répertoire de fichiers PDF

```python
from pipeline_indexation.pdf import PDFProcessor

processor = PDFProcessor()
results = processor.process_directory("path/to/your/pdf/directory")
print(f"Processed {len(results)} PDF files")
```

### Rechercher dans les documents indexés

```python
from pipeline_indexation.pdf import PDFProcessor

processor = PDFProcessor()
results = processor.search("your search query", n_results=5)
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Text: {result['text'][:100]}...")
    print(f"Metadata: {result['metadata']}")
```

## Personnalisation

Vous pouvez personnaliser le processeur PDF en modifiant les paramètres:

```python
processor = PDFProcessor(
    chroma_path="./custom_chroma_path",
    collection_name="custom_collection",
    chunk_size=500,
    chunk_overlap=50,
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

## Documentation du code

Pour plus de détails, consultez les docstrings dans le code source. 