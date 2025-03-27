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
- **Extraction intelligente des images et graphiques des PDF**
- **Détection et classification automatique des graphiques, figures et images**
- **Extraction des légendes des images et figures**
- **OCR avec Mistral pour une extraction de texte et d'images de haute qualité**
- Embedding des chunks avec HuggingFace
- Stockage dans une base de données ChromaDB

## Structure des fichiers

- `add_pdf.py` - Script pour ajouter un seul fichier PDF à la base de données
- `add_pdf_dir.py` - Script pour ajouter tous les fichiers PDF d'un répertoire
- `search.py` - Script pour rechercher dans la base de données
- `utils/pipeline_indexation/pdf.py` - Classe PDFProcessor implémentant la logique d'indexation
- `utils/mistral_ocr.py` - Module pour traiter les PDF avec Mistral OCR

## Utilisation

### Indexer un fichier PDF

Le script `add_pdf.py` vous permet d'ajouter des fichiers PDF à la base de données un par un :

```bash
python add_pdf.py chemin/vers/votre/document.pdf
```

Options disponibles :
```
--db-path PATH      Chemin vers la base de données Chroma (défaut: ./pdf_database)
--collection NAME   Nom de la collection dans Chroma (défaut: pdf_documents)
--chunk-size SIZE   Taille des chunks de texte (défaut: 1000)
--chunk-overlap SIZE Chevauchement entre les chunks (défaut: 200)
--model NAME        Modèle HuggingFace pour les embeddings
--no-images         Ne pas extraire les images du PDF
--image-min-size N  Taille minimale des images à extraire en pixels (défaut: 100)
--store-image-content Stocker le contenu des images en base64 dans les métadonnées
--use-mistral-ocr   Utiliser Mistral OCR pour le traitement du PDF
--mistral-api-key KEY Clé API pour Mistral AI (requise si --use-mistral-ocr est utilisé)
--image-output-dir PATH Répertoire pour sauvegarder les images extraites (défaut: ./images_extraites)
```

Exemple avec Mistral OCR :
```bash
python add_pdf.py document.pdf --use-mistral-ocr --mistral-api-key VOTRE_CLE_API
```

### Indexer un répertoire de fichiers PDF

Le script `add_pdf_dir.py` vous permet d'ajouter tous les fichiers PDF d'un répertoire à la base de données :

```bash
python add_pdf_dir.py chemin/vers/votre/repertoire
```

Options disponibles :
```
--db-path PATH      Chemin vers la base de données Chroma (défaut: ./pdf_database)
--collection NAME   Nom de la collection dans Chroma (défaut: pdf_documents)
--chunk-size SIZE   Taille des chunks de texte (défaut: 1000)
--chunk-overlap SIZE Chevauchement entre les chunks (défaut: 200)
--model NAME        Modèle HuggingFace pour les embeddings
--no-images         Ne pas extraire les images du PDF
--image-min-size N  Taille minimale des images à extraire en pixels (défaut: 100)
--store-image-content Stocker le contenu des images en base64 dans les métadonnées
--use-mistral-ocr   Utiliser Mistral OCR pour le traitement du PDF
--mistral-api-key KEY Clé API pour Mistral AI (requise si --use-mistral-ocr est utilisé)
--image-output-dir PATH Répertoire pour sauvegarder les images extraites (défaut: ./images_extraites)
```

Exemple avec Mistral OCR et un répertoire personnalisé pour les images :
```bash
python add_pdf_dir.py dossier_documents/ --use-mistral-ocr --mistral-api-key VOTRE_CLE_API --image-output-dir ./images_OCR
```

### Rechercher dans les documents indexés

Le script `search.py` vous permet de rechercher dans votre base de données de PDF :

```bash
python search.py "votre requête de recherche"
```

Options disponibles :
```
--db-path PATH      Chemin vers la base de données Chroma (défaut: ./pdf_database)
--collection NAME   Nom de la collection dans Chroma (défaut: pdf_documents)
--results N         Nombre de résultats à afficher (défaut: 5)
--model NAME        Modèle HuggingFace pour les embeddings
--filter KEY=VALUE  Filtrer les résultats (ex: content_type=image)
--graphs-only       Rechercher uniquement dans les graphiques et diagrammes
--images-only       Rechercher uniquement dans les images
--text-only         Rechercher uniquement dans le texte
```

Exemple avec des options personnalisées :
```bash
python search.py "intelligence artificielle" --db-path ./ma_base --collection mes_pdfs --results 10
```

Pour rechercher uniquement parmi les graphiques et diagrammes :
```bash
python search.py "évolution économique" --graphs-only
```

Pour rechercher uniquement parmi les images :
```bash
python search.py "diagramme financier" --images-only
```

Pour rechercher uniquement dans le texte :
```bash
python search.py "innovation technologique" --text-only
```

### Utilisation via l'API Python

#### Indexer un seul fichier PDF

```python
from utils.pipeline_indexation.pdf import PDFProcessor

# Avec PyMuPDF (par défaut)
processor = PDFProcessor(extract_images=True)
doc_ids = processor.process_pdf("chemin/vers/votre/document.pdf")
print(f"Ajout de {len(doc_ids)} éléments à la base de données")

# Avec Mistral OCR
processor = PDFProcessor(
    extract_images=True,
    use_mistral_ocr=True,
    mistral_api_key="VOTRE_CLE_API",
    image_output_dir="./images_extraites"
)
doc_ids = processor.process_pdf("chemin/vers/votre/document.pdf")
print(f"Ajout de {len(doc_ids)} éléments à la base de données")
```

#### Indexer un répertoire de fichiers PDF

```python
from utils.pipeline_indexation.pdf import PDFProcessor

# Avec PyMuPDF (par défaut)
processor = PDFProcessor(extract_images=True)
results = processor.process_directory("chemin/vers/votre/repertoire/pdf")
print(f"Traitement de {len(results)} fichiers PDF")

# Avec Mistral OCR
processor = PDFProcessor(
    extract_images=True,
    use_mistral_ocr=True,
    mistral_api_key="VOTRE_CLE_API",
    image_output_dir="./images_extraites"
)
results = processor.process_directory("chemin/vers/votre/repertoire/pdf")
print(f"Traitement de {len(results)} fichiers PDF")
```

#### Rechercher dans les documents indexés

```python
from utils.pipeline_indexation.pdf import PDFProcessor

processor = PDFProcessor()

# Recherche générale
results = processor.search("votre requête de recherche", n_results=5)

# Recherche uniquement dans les graphiques
graph_results = processor.search(
    "évolution économique", 
    n_results=5,
    filter_by={"type": "graph"}
)

# Recherche uniquement dans les images
image_results = processor.search(
    "diagramme financier", 
    n_results=5,
    filter_by={"content_type": "image"}
)

# Affichage des résultats
for i, result in enumerate(results):
    print(f"Résultat {i+1}:")
    print(f"Source: {result['metadata'].get('filename', 'Inconnu')}")
    print(f"Titre: {result['metadata'].get('title', 'Inconnu')}")
    print(f"Texte: {result['text'][:100]}...")
```

## Traitement des PDF avec Mistral OCR

Le système propose désormais une intégration avec Mistral OCR pour une extraction de texte et d'images de haute qualité.

### Avantages de Mistral OCR

1. **Qualité d'extraction supérieure** - Mistral OCR offre une reconnaissance de texte de pointe, particulièrement efficace pour les documents complexes ou scannés.
2. **Détection avancée des images** - Extraction précise des images même dans des layouts complexes.
3. **Classification intelligente** - Détection automatique des figures, tableaux et graphiques.
4. **Extraction de légendes** - Identifie automatiquement les légendes associées aux images.
5. **Analyses multimodales** - Comprend le lien entre le texte et les éléments visuels.

### Utilisation de Mistral OCR

Pour utiliser Mistral OCR, vous devez d'abord obtenir une clé API auprès de Mistral AI. Une fois la clé obtenue, vous pouvez utiliser Mistral OCR de deux façons :

1. **Via la ligne de commande** :
   ```bash
   python add_pdf.py document.pdf --use-mistral-ocr --mistral-api-key VOTRE_CLE_API
   ```

2. **Via une variable d'environnement** :
   ```bash
   export MISTRAL_API_KEY="VOTRE_CLE_API"
   python add_pdf.py document.pdf --use-mistral-ocr
   ```

3. **Dans votre code Python** :
   ```python
   processor = PDFProcessor(
       use_mistral_ocr=True,
       mistral_api_key="VOTRE_CLE_API",
       image_output_dir="./images_extraites"
   )
   ```

Les images extraites sont automatiquement sauvegardées dans le répertoire spécifié par `--image-output-dir` (par défaut : `./images_extraites`). Chaque document PDF a son propre sous-répertoire pour faciliter l'organisation.

## Traitement amélioré des graphiques et figures

Le système intègre maintenant des algorithmes améliorés pour détecter et classifier les images :

1. **Détection avancée d'images** - Utilise multiple méthodes pour extraire les images même lorsque la méthode standard échoue.
2. **Classification intelligente** - Analyse les images pour déterminer s'il s'agit de graphiques, figures ou images simples.
3. **Extraction de légendes** - Tente d'extraire automatiquement les légendes associées aux images.
4. **Analyse de contexte** - Utilise le texte environnant pour mieux comprendre le contenu de l'image.
5. **Filtrage amélioré** - Permet de rechercher spécifiquement des graphiques avec l'option `--graphs-only`.

### Méthodes de détection d'images

Le système utilise plusieurs méthodes pour maximiser la détection d'images :

1. **PyMuPDF (standard)** - Utilise `page.get_images()` et `doc.extract_image()`.
2. **PyMuPDF (alternative)** - Utilise `page.get_text("dict")` pour trouver les images non détectées par la méthode standard.
3. **Mistral OCR** - Utilise les capacités avancées de l'API Mistral OCR pour détecter les images et leur contexte.
4. **Analyse d'image** - Utilise des heuristiques basées sur le nombre de couleurs et la détection de bords pour classifier les graphiques.

Pour désactiver l'extraction d'images :
```bash
python add_pdf.py document.pdf --no-images
```

## Personnalisation

Vous pouvez personnaliser le processeur PDF en modifiant les paramètres:

```python
processor = PDFProcessor(
    chroma_path="./custom_chroma_path",
    collection_name="custom_collection",
    chunk_size=500,
    chunk_overlap=50,
    model_name="sentence-transformers/all-mpnet-base-v2",
    extract_images=True,
    image_min_size=200,  # Taille minimale en pixels
    store_image_content=False,  # Ne pas stocker le contenu des images
    use_mistral_ocr=True,  # Utiliser Mistral OCR
    mistral_api_key="VOTRE_CLE_API",  # Clé API Mistral
    image_output_dir="./images_extraites"  # Répertoire pour les images
)
```

## Workflow recommandé

1. **Ajoutez des documents individuels** au fur et à mesure que vous les recevez :
   ```bash
   python add_pdf.py nouveau_document.pdf
   ```
   Ou avec Mistral OCR pour une meilleure qualité :
   ```bash
   python add_pdf.py nouveau_document.pdf --use-mistral-ocr --mistral-api-key VOTRE_CLE_API
   ```

2. **Ajoutez des collections entières** de documents quand vous avez un lot à traiter :
   ```bash
   python add_pdf_dir.py dossier_documents/
   ```
   Ou avec Mistral OCR :
   ```bash
   python add_pdf_dir.py dossier_documents/ --use-mistral-ocr --mistral-api-key VOTRE_CLE_API
   ```

3. **Recherchez dans votre base de connaissances** :
   ```bash
   python search.py "votre question ou requête"
   ```
   
4. **Recherchez spécifiquement dans les graphiques et diagrammes** :
   ```bash
   python search.py "évolution économique" --graphs-only
   ```

## Documentation du code

Pour plus de détails, consultez les docstrings dans le code source. 