# Analyseur de PDF avec Mistral OCR pour RAG

## Description
Ce programme analyse les documents PDF pour en extraire le texte, les images et les tableaux en utilisant l'API Mistral OCR. Il génère ensuite des embeddings pour ces contenus et les stocke dans une base de données vectorielle, prêt à être utilisé dans un système de Retrieval-Augmented Generation (RAG). Il peut également exporter tous les éléments extraits dans un dossier local organisé.

## Fonctionnalités
- Extraction complète du texte avec découpage intelligent en chunks
- Détection et sauvegarde des images
- Détection spécifique des tableaux
- Tentative de récupération des légendes pour les images et tableaux
- Extraction de métadonnées (titre, auteur, date)
- Génération d'embeddings pour tous les contenus
- Stockage dans une base de données vectorielle (ChromaDB)
- Organisation claire des résultats (texte, images, tableaux)
- **Nouveau:** Exportation des éléments extraits dans un dossier local
- **Nouveau:** Options en ligne de commande pour plus de flexibilité

## Prérequis
Pour utiliser ce programme, vous devez installer les dépendances suivantes:
```
pip install mistralai pillow datauri sentence-transformers langchain chromadb
```

## Configuration
1. Modifiez la clé API Mistral dans le script:
```python
API_KEY = "votre_clé_api_mistral"
```

2. Configurez les paramètres selon vos besoins:
```python
# Dossier pour les images extraites
IMAGE_DIR = "images_extraites"

# Dossier pour l'exportation des éléments
EXPORT_BASE_DIR = "./data/output"

# Paramètres pour le découpage du texte
CHUNK_SIZE = 1000     # Taille de chaque morceau 
CHUNK_OVERLAP = 200   # Chevauchement entre les morceaux

# Modèle d'embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chemin de la base de données
DB_PATH = "./vectordb"
```

## Utilisation
### Ligne de commande

Le script supporte désormais plusieurs modes d'utilisation :

1. **Analyse et exportation (mode par défaut)** :
   ```
   python mistral_pdf_analyzer.py chemin/vers/votre/document.pdf
   ```
   Cette commande va à la fois analyser le PDF, stocker les données dans ChromaDB et exporter les éléments dans un dossier.

2. **Analyse sans exportation** :
   ```
   python mistral_pdf_analyzer.py chemin/vers/votre/document.pdf --no-export
   ```
   Cette commande analyse le PDF et stocke les données dans ChromaDB sans les exporter.

3. **Exportation uniquement** :
   ```
   python mistral_pdf_analyzer.py chemin/vers/votre/document.pdf --export-only
   ```
   Cette commande exporte les éléments d'un PDF déjà analysé, sans refaire l'analyse.

4. **Utilisation par défaut** :
   ```
   python mistral_pdf_analyzer.py
   ```
   Sans arguments, le script utilisera le PDF par défaut configuré dans le code.

### Structure des fichiers exportés

Les éléments exportés sont organisés dans un dossier daté, avec la structure suivante :
```
data/output/nom_du_fichier_date_heure/
  ├── texte/                    # Dossier contenant les chunks de texte individuels
  │   ├── chunk_000.txt
  │   ├── chunk_001.txt
  │   └── ...
  ├── images/                   # Dossier contenant les images extraites
  │   ├── page1_image0.jpeg
  │   ├── page1_image0.jpeg.txt # Métadonnées de l'image
  │   └── ...
  ├── tableaux/                 # Dossier contenant les tableaux extraits
  │   ├── page2_tableau0.jpeg
  │   ├── page2_tableau0.jpeg.txt # Métadonnées du tableau
  │   └── ...
  ├── texte_complet.txt         # Fichier contenant tout le texte
  └── resume.txt                # Résumé de l'exportation
```

## Structure du code
Le code est organisé en sections claires:
1. **Imports et configuration** - Bibliothèques et paramètres
2. **Fonctions principales** - Structure générale du processus
3. **Fonction d'exportation** - Exportation des éléments dans un dossier local
4. **Extraction de contenu** - Fonctions pour extraire texte, images, tableaux et métadonnées
5. **Traitement des données** - Découpage du texte et génération d'embeddings
6. **Stockage vectoriel** - Intégration avec ChromaDB
7. **Fonctions auxiliaires** - Utilitaires pour le traitement des contenus
8. **Point d'entrée principal** - Exécution du programme avec gestion des arguments

## Utilisation dans un système RAG
Une fois le traitement terminé, vous pouvez utiliser la base vectorielle pour:
1. Rechercher du contenu similaire à une requête:
```python
import chromadb

# Connexion à la base
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection("pdf_collection")

# Recherche (exemple)
resultat = collection.query(
    query_texts=["ma question sur le document"],
    n_results=5
)

# Accéder aux résultats
for doc in resultat["documents"][0]:
    print(doc)
```

2. Filtrer par type de contenu:
```python
# Recherche uniquement dans le texte
resultat = collection.query(
    query_texts=["ma question"],
    where={"type": "text"},
    n_results=3
)

# Ou uniquement dans les tableaux
resultat = collection.query(
    query_texts=["statistiques annuelles"],
    where={"type": "table"},
    n_results=3
)
```

## Exemple complet
```python
from mistral_pdf_analyzer import analyser_pdf
import chromadb

# Analyser un PDF
resultat = analyser_pdf("mon_document.pdf")
print(f"Éléments indexés: {len(resultat['ids_stockes'])}")

# Rechercher dans la base
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection("pdf_collection")
resultats = collection.query(
    query_texts=["résumé des points clés"],
    n_results=3
)

# Afficher les résultats
for doc in resultats["documents"][0]:
    print("---")
    print(doc)
``` 

## Travail avec les fichiers exportés
Une fois les fichiers exportés, vous pouvez les utiliser de plusieurs façons :

1. **Consultation directe** : Tous les éléments sont exportés dans un format facilement consultable (texte, images JPEG)

2. **Importation dans d'autres outils** : Les fichiers texte peuvent être importés dans d'autres outils d'analyse

3. **Archivage** : Conservez une copie locale des éléments extraits pour consultation ultérieure, même sans accès à la base de données vectorielle 