# Analyseur de PDF avec Mistral OCR pour RAG

## Description
Ce programme est un processeur de documents PDF qui utilise l'API Mistral OCR pour extraire et structurer le contenu. Il est spécialement conçu pour préparer des documents à utiliser dans un système RAG (Retrieval-Augmented Generation). Le processeur extrait intelligemment le texte, les images et les tableaux, génère des embeddings pour chaque élément, et stocke le tout dans une base de données vectorielle.

## Fonctionnalités
- Extraction intelligente du texte avec OCR via Mistral
- Détection et extraction des images avec exclusion automatique des tableaux
- Détection avancée des tableaux (via propriété native et fallback par regex)
- Extraction automatique des légendes et du contexte pour les images
- Nettoyage et structuration des tableaux en format Markdown
- Extraction des métadonnées (titre, auteur, date) depuis les premières pages
- Découpage intelligent du texte en chunks avec chevauchement configurable
- Génération d'embeddings pour tous les types de contenus
- Stockage optimisé dans ChromaDB avec métadonnées enrichies
- Export structuré des éléments dans un dossier daté

## Prérequis
```bash
pip install mistralai pillow datauri sentence-transformers langchain chromadb
```

## Configuration
1. Configurez votre clé API Mistral dans une variable d'environnement :
```bash
export MISTRAL_API_KEY="votre_clé_api"
```

2. Les principaux paramètres sont configurables en début de fichier :
```python
# Dossiers de stockage
IMAGE_DIR = "data/output/images"
TABLE_DIR = "data/output/tables"
EXPORT_BASE_DIR = "data/output"

# Paramètres de découpage
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Modèle d'embeddings et stockage
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "data/vectordb"
```

## Utilisation
### En ligne de commande

```bash
# Analyse complète avec export
python pdf_processor.py /chemin/vers/document.pdf

# Mode interactif
python pdf_processor.py
```

### En tant que module

```python
from utils.pipeline_indexation.pdf_processor import analyser_pdf

# Analyse avec export
resultat = analyser_pdf("document.pdf", exporter=True)

# Analyse sans export
resultat = analyser_pdf("document.pdf", exporter=False)
```

### Structure des fichiers exportés

```
data/output/nom_fichier_YYYYMMDD_HHMMSS/
├── texte/
│   ├── texte_complet.txt     # Texte intégral
│   ├── chunk_000.txt         # Chunks individuels
│   └── ...
├── images/
│   ├── pageX_imageY.jpeg     # Images extraites
│   └── ...
└── tableaux/
    ├── pageX_tableauY.txt    # Tableaux en format Markdown
    └── ...
```

## Architecture du code
Le processeur est organisé en sections fonctionnelles distinctes :

1. **Configuration et initialisation**
   - Configuration des chemins et paramètres
   - Initialisation des dossiers nécessaires

2. **Fonction principale `analyser_pdf()`**
   - Orchestre le processus complet d'analyse
   - Gère les différentes étapes d'extraction et de traitement

3. **Extraction de contenu**
   - `traiter_pdf_avec_mistral()` : OCR via API Mistral
   - `extraire_texte_propre()` : Extraction du texte brut
   - `extraire_images_sans_tableaux()` : Gestion des images
   - `detecter_tableaux()` : Détection multi-méthodes des tableaux

4. **Traitement avancé**
   - `clean_table_structure()` : Nettoyage des tableaux
   - `extraire_metadonnees()` : Extraction des métadonnées
   - `decouper_texte()` : Découpage intelligent du texte

5. **Génération d'embeddings**
   - Fonctions spécialisées pour texte, images et tableaux
   - Gestion du contexte et des descriptions

6. **Stockage et export**
   - `stocker_dans_base_vectorielle()` : Intégration ChromaDB
   - `exporter_elements()` : Export structuré des contenus

## Utilisation dans un système RAG
La base vectorielle peut être interrogée efficacement :

```python
import chromadb

client = chromadb.PersistentClient(path="data/vectordb")
collection = client.get_collection("pdf_collection")

# Recherche par type de contenu
resultats = collection.query(
    query_texts=["ma recherche"],
    where={"type": "text"},  # ou "image" ou "table"
    n_results=3
)

# Affichage des résultats
for doc, meta in zip(resultats["documents"][0], resultats["metadatas"][0]):
    print(f"Type: {meta['type']}")
    print(f"Source: Page {meta.get('page', 'N/A')}")
    print(f"Contenu: {doc}\n")
```