"""
ANALYSEUR PDF AVEC MISTRAL OCR POUR RAG
---------------------------------------
Ce script permet d'extraire le texte, les images et les tableaux d'un document PDF,
de générer des embeddings, de stocker ces données dans une base vectorielle,
et d'exporter les éléments dans un dossier local.
"""

# -------- IMPORTS --------
import os
import re
import sys
import shutil
from io import BytesIO
from typing import List, Dict, Any
import hashlib
from datetime import datetime

# Pour l'analyse de PDF
from PIL import Image
import datauri
from mistralai import Mistral

# Pour les embeddings et la base de données
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------- CONFIGURATION --------
# Votre clé API Mistral
# On a enregistré la clé API dans une variable d'environnement pour la sécurité (avec export MISTRAL_API_KEY=...)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Dossier où les images extraites seront sauvegardées
IMAGE_DIR = "images_extraites"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Dossier d'exportation des éléments
EXPORT_BASE_DIR = "./data/output"
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

# Paramètres pour le découpage du texte
CHUNK_SIZE = 1000     # Taille de chaque morceau de texte
CHUNK_OVERLAP = 200   # Chevauchement entre les morceaux

# Modèle d'embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modèle léger et rapide

# Base de données vectorielle
DB_PATH = "./vectordb"

# -------- FONCTIONS PRINCIPALES --------

def analyser_pdf(chemin_pdf, exporter=True):
    """
    Fonction principale qui analyse un PDF et l'intègre dans une base vectorielle.
    
    Args:
        chemin_pdf: Chemin vers le fichier PDF à analyser
        exporter: Si True, exporte également les éléments dans un dossier local
    
    Returns:
        dict: Résumé du processus d'indexation
    """
    print(f"Analyse du PDF: {chemin_pdf}")
    
    # Étape 1: Initialiser le client Mistral
    client = Mistral(api_key=API_KEY)
    
    # Étape 2: Extraire le contenu du PDF (texte, images, tableaux, métadonnées)
    resultat_ocr = traiter_pdf_avec_mistral(client, chemin_pdf)
    texte = extraire_texte_propre(resultat_ocr)
    images = extraire_et_sauvegarder_images(resultat_ocr)
    tableaux = detecter_tableaux(resultat_ocr)
    metadonnees = extraire_metadonnees(resultat_ocr, chemin_pdf)
    
    # Étape 3: Découper le texte en chunks
    chunks = decouper_texte(texte)
    
    # Étape 4: Initialiser le modèle d'embeddings
    modele_embeddings = SentenceTransformer(EMBEDDING_MODEL)
    
    # Étape 5: Générer les embeddings
    embeddings_texte = generer_embeddings(modele_embeddings, chunks)
    embeddings_images = generer_embeddings_images(modele_embeddings, images)
    embeddings_tableaux = generer_embeddings_images(modele_embeddings, tableaux, est_tableau=True)
    
    # Étape 6: Stocker dans la base vectorielle
    stockage_db = stocker_dans_base_vectorielle(
        chemin_pdf, 
        chunks, embeddings_texte, 
        images, embeddings_images,
        tableaux, embeddings_tableaux,
        metadonnees
    )
    
    # Résumé des résultats
    resume = {
        "fichier": chemin_pdf,
        "chunks_texte": len(chunks),
        "images": len(images),
        "tableaux": len(tableaux),
        "ids_stockes": stockage_db
    }
    
    print(f"Analyse terminée: {len(chunks)} chunks, {len(images)} images, {len(tableaux)} tableaux")
    
    # Étape 7 (optionnelle): Exporter les éléments dans un dossier local
    if exporter:
        dossier_export = exporter_elements(chemin_pdf)
        resume["dossier_export"] = dossier_export
    
    return resume

# -------- FONCTION D'EXPORTATION --------

def exporter_elements(chemin_pdf, export_dir=None):
    """
    Exporte tous les éléments liés à un PDF (texte, images, tableaux) dans un dossier.
    
    Args:
        chemin_pdf: Chemin du fichier PDF analysé
        export_dir: Dossier d'exportation (optionnel, généré si non fourni)
    
    Returns:
        Le chemin du dossier d'exportation
    """
    # Générer l'ID du document
    pdf_id = hashlib.md5(chemin_pdf.encode()).hexdigest()
    nom_fichier = os.path.basename(chemin_pdf).replace('.pdf', '')
    
    # Créer un dossier d'exportation avec date/heure
    if not export_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(EXPORT_BASE_DIR, f"{nom_fichier}_{timestamp}")
    
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(os.path.join(export_dir, "texte"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "tableaux"), exist_ok=True)
    
    print(f"Exportation des éléments pour: {chemin_pdf}")
    print(f"Dossier d'exportation: {export_dir}")
    
    try:
        # Initialiser la connexion à ChromaDB
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("pdf_collection")
        
        # Récupérer tous les éléments liés au document
        resultats_texte = collection.get(
            where={"document_id": {"$eq": pdf_id}},
            include=["documents", "metadatas"]
        )
        
        # Filtrer pour ne garder que les éléments de type "text"
        documents_texte = []
        metadatas_texte = []
        for i, meta in enumerate(resultats_texte["metadatas"]):
            if meta.get("type") == "text":
                documents_texte.append(resultats_texte["documents"][i])
                metadatas_texte.append(meta)
        
        print(f"Exportation de {len(documents_texte)} chunks de texte...")
        for i, (doc, meta) in enumerate(zip(documents_texte, metadatas_texte)):
            # Créer un fichier texte pour chaque chunk
            fichier_texte = os.path.join(export_dir, "texte", f"chunk_{meta.get('chunk_index', i):03d}.txt")
            with open(fichier_texte, "w", encoding="utf-8") as f:
                f.write(f"--- CHUNK {meta.get('chunk_index', i)}/{meta.get('chunk_count', 1)-1} ---\n\n")
                f.write(doc)
        
        # Exportation du texte complet dans un seul fichier
        with open(os.path.join(export_dir, "texte_complet.txt"), "w", encoding="utf-8") as f:
            for i, doc in enumerate(sorted(zip(documents_texte, metadatas_texte), 
                                         key=lambda x: x[1].get('chunk_index', 0))):
                f.write(f"--- CHUNK {doc[1].get('chunk_index', i)}/{doc[1].get('chunk_count', 1)-1} ---\n\n")
                f.write(doc[0])
                f.write("\n\n" + "-"*50 + "\n\n")
        
        # Exporter les images (filtrer les résultats déjà obtenus)
        documents_images = []
        metadatas_images = []
        for i, meta in enumerate(resultats_texte["metadatas"]):
            if meta.get("type") == "image":
                documents_images.append(resultats_texte["documents"][i])
                metadatas_images.append(meta)
        
        print(f"Exportation de {len(documents_images)} images...")
        for i, (doc, meta) in enumerate(zip(documents_images, metadatas_images)):
            # Copier l'image dans le dossier d'exportation
            if os.path.exists(meta.get('image_path', '')):
                fichier_dest = os.path.join(export_dir, "images", os.path.basename(meta.get('image_path')))
                shutil.copy2(meta.get('image_path'), fichier_dest)
                
                # Créer un fichier texte avec les métadonnées
                fichier_meta = os.path.join(export_dir, "images", f"{os.path.basename(meta.get('image_path'))}.txt")
                with open(fichier_meta, "w", encoding="utf-8") as f:
                    f.write(f"Image: {os.path.basename(meta.get('image_path'))}\n")
                    f.write(f"Page: {meta.get('page')}\n")
                    if meta.get('caption'):
                        f.write(f"Légende: {meta.get('caption')}\n")
                    f.write(f"Taille: {meta.get('width')}x{meta.get('height')}\n")
        
        # Exporter les tableaux (filtrer les résultats déjà obtenus)
        documents_tableaux = []
        metadatas_tableaux = []
        for i, meta in enumerate(resultats_texte["metadatas"]):
            if meta.get("type") == "table":
                documents_tableaux.append(resultats_texte["documents"][i])
                metadatas_tableaux.append(meta)
        
        print(f"Exportation de {len(documents_tableaux)} tableaux...")
        for i, (doc, meta) in enumerate(zip(documents_tableaux, metadatas_tableaux)):
            # Copier le tableau dans le dossier d'exportation
            if os.path.exists(meta.get('table_path', '')):
                fichier_dest = os.path.join(export_dir, "tableaux", os.path.basename(meta.get('table_path')))
                shutil.copy2(meta.get('table_path'), fichier_dest)
                
                # Créer un fichier texte avec les métadonnées
                fichier_meta = os.path.join(export_dir, "tableaux", f"{os.path.basename(meta.get('table_path'))}.txt")
                with open(fichier_meta, "w", encoding="utf-8") as f:
                    f.write(f"Tableau: {os.path.basename(meta.get('table_path'))}\n")
                    f.write(f"Page: {meta.get('page')}\n")
                    if meta.get('caption'):
                        f.write(f"Légende: {meta.get('caption')}\n")
                    f.write(f"Taille: {meta.get('width')}x{meta.get('height')}\n")
        
        # Créer un fichier de résumé
        with open(os.path.join(export_dir, "resume.txt"), "w", encoding="utf-8") as f:
            f.write(f"RÉSUMÉ DE L'EXPORTATION\n")
            f.write(f"======================\n\n")
            f.write(f"Document: {chemin_pdf}\n")
            f.write(f"ID document: {pdf_id}\n")
            f.write(f"Date d'exportation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Nombre de chunks de texte: {len(documents_texte)}\n")
            f.write(f"Nombre d'images: {len(documents_images)}\n")
            f.write(f"Nombre de tableaux: {len(documents_tableaux)}\n")
        
        print(f"Exportation terminée dans: {export_dir}")
        return export_dir
    
    except Exception as e:
        print(f"Erreur lors de l'exportation: {str(e)}")
        return None
        
# -------- EXTRACTION DE CONTENU --------

def traiter_pdf_avec_mistral(client, chemin_pdf):
    """
    Envoie le PDF à l'API Mistral OCR pour traitement.
    
    Args:
        client: Instance du client Mistral
        chemin_pdf: Chemin vers le fichier PDF
        
    Returns:
        Réponse de l'API Mistral OCR
    """
    print("Envoi du PDF à Mistral OCR...")
    
    # Téléchargement du fichier
    fichier_telecharge = client.files.upload(
        file={"file_name": chemin_pdf, "content": open(chemin_pdf, "rb")},
        purpose="ocr"
    )
    
    # Obtention de l'URL signée
    url_signee = client.files.get_signed_url(file_id=fichier_telecharge.id)
    
    # Traitement OCR
    resultat_ocr = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": url_signee.url},
        include_image_base64=True
    )
    
    print("Traitement OCR terminé")
    return resultat_ocr

def extraire_texte_propre(resultat_ocr):
    """
    Extrait le texte du résultat OCR et le nettoie.
    
    Args:
        resultat_ocr: Réponse de l'API Mistral OCR
        
    Returns:
        Texte extrait nettoyé
    """
    print("Extraction du texte...")
    
    # Fusionner le texte markdown de toutes les pages
    texte_brut = ""
    for page in resultat_ocr.pages:
        if page.markdown:
            # Supprimer les références aux images
            texte_page = re.sub(r'!\[.*?\]\(.*?\)', '', page.markdown)
            texte_brut += texte_page + "\n\n"
    
    return texte_brut

def extraire_et_sauvegarder_images(resultat_ocr):
    """
    Extrait et sauvegarde les images du résultat OCR.
    
    Args:
        resultat_ocr: Réponse de l'API Mistral OCR
        
    Returns:
        Liste de dictionnaires avec les informations sur les images
    """
    print("Extraction des images...")
    images_info = []
    
    # Parcourir toutes les pages
    for numero_page, page in enumerate(resultat_ocr.pages, 1):
        if not page.images:
            continue
            
        # Traiter chaque image de la page
        for i, image in enumerate(page.images):
            try:
                # Vérifier si c'est un tableau (on le traitera séparément)
                if est_probablement_tableau(image, page.markdown):
                    continue
                
                # Convertir base64 en image
                donnees_image = datauri.parse(image.image_base64).data
                image_pil = Image.open(BytesIO(donnees_image))
                
                # Générer nom de fichier unique
                nom_fichier = f"page{numero_page}_image{i}.jpeg"
                chemin_image = os.path.join(IMAGE_DIR, nom_fichier)
                
                # Sauvegarder l'image
                image_pil.save(chemin_image)
                
                # Récupérer la légende ou le contexte
                legende = trouver_legende_image(image, page.markdown)
                
                # Ajouter aux informations
                images_info.append({
                    "page": numero_page,
                    "chemin": chemin_image,
                    "id": image.id,
                    "largeur": image_pil.width,
                    "hauteur": image_pil.height,
                    "legende": legende,
                    "type": "image",
                    "contexte": extraire_contexte(image, page.markdown)
                })
                
                print(f"  Image extraite: {chemin_image}")
            except Exception as e:
                print(f"  Erreur lors du traitement d'une image: {str(e)}")
    
    return images_info

def detecter_tableaux(resultat_ocr):
    """
    Détecte les tableaux dans le résultat OCR.
    
    Args:
        resultat_ocr: Réponse de l'API Mistral OCR
        
    Returns:
        Liste de dictionnaires avec les informations sur les tableaux
    """
    print("Détection des tableaux...")
    tableaux_info = []
    
    # Parcourir toutes les pages
    for numero_page, page in enumerate(resultat_ocr.pages, 1):
        if not page.images:
            continue
            
        # Traiter chaque image de la page
        for i, image in enumerate(page.images):
            try:
                # Vérifier si c'est un tableau
                if not est_probablement_tableau(image, page.markdown):
                    continue
                    
                # Convertir base64 en image
                donnees_image = datauri.parse(image.image_base64).data
                image_pil = Image.open(BytesIO(donnees_image))
                
                # Générer nom de fichier unique
                nom_fichier = f"page{numero_page}_tableau{i}.jpeg"
                chemin_image = os.path.join(IMAGE_DIR, nom_fichier)
                
                # Sauvegarder l'image du tableau
                image_pil.save(chemin_image)
                
                # Récupérer la légende ou le contexte
                legende = trouver_legende_image(image, page.markdown)
                
                # Ajouter aux informations
                tableaux_info.append({
                    "page": numero_page,
                    "chemin": chemin_image,
                    "id": image.id,
                    "largeur": image_pil.width,
                    "hauteur": image_pil.height,
                    "legende": legende,
                    "type": "tableau",
                    "contexte": extraire_contexte(image, page.markdown)
                })
                
                print(f"  Tableau détecté: {chemin_image}")
            except Exception as e:
                print(f"  Erreur lors du traitement d'un tableau: {str(e)}")
    
    return tableaux_info

def extraire_metadonnees(resultat_ocr, chemin_pdf):
    """
    Extrait les métadonnées du document.
    
    Args:
        resultat_ocr: Réponse de l'API Mistral OCR
        chemin_pdf: Chemin du fichier PDF
        
    Returns:
        Dictionnaire des métadonnées
    """
    print("Extraction des métadonnées...")
    
    # Métadonnées de base
    metadonnees = {
        "nom_fichier": os.path.basename(chemin_pdf),
        "chemin": chemin_pdf,
        "nombre_pages": len(resultat_ocr.pages)
    }
    
    # Texte des premières pages pour trouver des métadonnées
    texte_debut = ""
    for i, page in enumerate(resultat_ocr.pages):
        if i < 2 and page.markdown:  # Utiliser seulement les 2 premières pages
            texte_debut += page.markdown + "\n"
    
    # Extraire titre
    titre_patterns = [
        r"# (.*?)[\n\r]",  # Titre formaté en markdown
        r"^([^\n\r]{5,100})[\n\r]",  # Première ligne
        r"(?:Title|Titre)[\s:]+([^\n\r]{5,100})[\n\r]"  # Motif explicite
    ]
    
    for pattern in titre_patterns:
        match = re.search(pattern, texte_debut, re.MULTILINE)
        if match:
            metadonnees["titre"] = match.group(1).strip()
            break
    
    # Extraire auteur
    auteur_patterns = [
        r"(?:Author|Auteur)[s]?[\s:]+([^\n\r]{3,100})[\n\r]",
        r"(?:By|Par)[\s:]+([^\n\r]{3,100})[\n\r]"
    ]
    
    for pattern in auteur_patterns:
        match = re.search(pattern, texte_debut, re.MULTILINE | re.IGNORECASE)
        if match:
            metadonnees["auteur"] = match.group(1).strip()
            break
    
    # Extraire date
    date_patterns = [
        r"(?:Date|Date de publication)[\s:]+([^\n\r]{3,30})[\n\r]",
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre) \d{1,2},? \d{4})\b"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, texte_debut, re.MULTILINE | re.IGNORECASE)
        if match:
            metadonnees["date"] = match.group(1).strip()
            break
    
    return metadonnees

# -------- TRAITEMENT DES DONNÉES --------

def decouper_texte(texte):
    """
    Découpe le texte en chunks pour le traitement.
    
    Args:
        texte: Texte complet à découper
        
    Returns:
        Liste de chunks de texte
    """
    print(f"Découpage du texte en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
    
    # Utilisation du RecursiveCharacterTextSplitter pour un découpage intelligent
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(texte)
    print(f"  Texte découpé en {len(chunks)} chunks")
    
    return chunks

def generer_embeddings(modele, textes):
    """
    Génère des embeddings pour une liste de textes.
    
    Args:
        modele: Modèle SentenceTransformer
        textes: Liste de textes
        
    Returns:
        Liste d'embeddings
    """
    print(f"Génération d'embeddings pour {len(textes)} textes...")
    return modele.encode(textes, show_progress_bar=False)

def generer_embeddings_images(modele, elements, est_tableau=False):
    """
    Génère des embeddings pour des images ou tableaux en utilisant leur légende et contexte.
    
    Args:
        modele: Modèle SentenceTransformer
        elements: Liste d'informations sur les images ou tableaux
        est_tableau: Indique si ce sont des tableaux
        
    Returns:
        Liste d'embeddings
    """
    type_element = "tableaux" if est_tableau else "images"
    print(f"Génération d'embeddings pour {len(elements)} {type_element}...")
    
    # Créer des descriptions textuelles pour chaque élément
    descriptions = []
    for el in elements:
        description = ""
        if est_tableau:
            description += f"Tableau à la page {el['page']}. "
        else:
            description += f"Image à la page {el['page']}. "
            
        if el.get('legende'):
            description += f"Légende: {el['legende']}. "
            
        if el.get('contexte'):
            description += f"Contexte: {el['contexte'][:200]}"
            
        descriptions.append(description)
    
    if not descriptions:
        return []
        
    return modele.encode(descriptions, show_progress_bar=False)

def stocker_dans_base_vectorielle(
    chemin_pdf, chunks, embeddings_texte, 
    images, embeddings_images, 
    tableaux, embeddings_tableaux,
    metadonnees
):
    """
    Stocke toutes les données dans une base vectorielle Chroma.
    
    Args:
        chemin_pdf: Chemin du fichier PDF
        chunks: Liste des chunks de texte
        embeddings_texte: Embeddings pour les chunks
        images: Liste des infos sur les images
        embeddings_images: Embeddings pour les images
        tableaux: Liste des infos sur les tableaux
        embeddings_tableaux: Embeddings pour les tableaux
        metadonnees: Métadonnées du document
        
    Returns:
        Liste des IDs stockés
    """
    print(f"Stockage dans la base vectorielle...")
    
    # Initialiser la base de données
    client = chromadb.PersistentClient(path=DB_PATH)
    collection_name = "pdf_collection"
    
    # Vérifier si la collection existe, sinon la créer
    try:
        collection = client.get_collection(collection_name)
        print(f"  Collection '{collection_name}' trouvée")
    except:
        collection = client.create_collection(collection_name)
        print(f"  Collection '{collection_name}' créée")
    
    # Générer un ID unique pour le document
    pdf_id = hashlib.md5(chemin_pdf.encode()).hexdigest()
    nom_fichier = os.path.basename(chemin_pdf)
    
    ids_stockes = []
    
    # Stocker les chunks de texte
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_texte)):
        doc_id = f"{pdf_id}_txt_{i}"
        ids_stockes.append(doc_id)
        
        # Métadonnées pour ce chunk
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "type": "text",
            "chunk_index": i,
            "chunk_count": len(chunks)
        }
        
        # Ajouter les métadonnées du document
        meta.update({f"doc_{k}": v for k, v in metadonnees.items()})
        
        # Ajouter à la collection
        collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[meta],
            documents=[chunk]
        )
    
    # Stocker les images
    for i, (img, embedding) in enumerate(zip(images, embeddings_images)):
        doc_id = f"{pdf_id}_img_{i}"
        ids_stockes.append(doc_id)
        
        # Métadonnées pour cette image
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "type": "image",
            "page": img["page"],
            "image_path": img["chemin"],
            "width": img["largeur"],
            "height": img["hauteur"]
        }
        
        # Ajouter la légende si disponible
        if img.get("legende"):
            meta["caption"] = img["legende"]
            
        # Ajouter à la collection
        description = f"Image à la page {img['page']}"
        if img.get("legende"):
            description += f": {img['legende']}"
            
        collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[meta],
            documents=[description]
        )
    
    # Stocker les tableaux
    for i, (tab, embedding) in enumerate(zip(tableaux, embeddings_tableaux)):
        doc_id = f"{pdf_id}_tab_{i}"
        ids_stockes.append(doc_id)
        
        # Métadonnées pour ce tableau
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "type": "table",
            "page": tab["page"],
            "table_path": tab["chemin"],
            "width": tab["largeur"],
            "height": tab["hauteur"]
        }
        
        # Ajouter la légende si disponible
        if tab.get("legende"):
            meta["caption"] = tab["legende"]
            
        # Ajouter à la collection
        description = f"Tableau à la page {tab['page']}"
        if tab.get("legende"):
            description += f": {tab['legende']}"
            
        collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[meta],
            documents=[description]
        )
    
    print(f"  {len(ids_stockes)} éléments stockés dans la base vectorielle")
    return ids_stockes

# -------- FONCTIONS AUXILIAIRES --------

def est_probablement_tableau(image, texte_markdown):
    """
    Détermine si une image est probablement un tableau.
    
    Args:
        image: Objet image du résultat OCR
        texte_markdown: Texte markdown contenant potentiellement des indices
        
    Returns:
        Boolean indiquant si c'est probablement un tableau
    """
    # Rechercher des indices dans le texte
    tableau_indices = ["tableau", "table", "tabular", "grid"]
    
    # Chercher la légende de l'image
    legende = trouver_legende_image(image, texte_markdown)
    
    # Chercher dans le contexte
    contexte = extraire_contexte(image, texte_markdown)
    
    # Vérifier si les indices de tableau apparaissent dans la légende ou le contexte
    texte_a_verifier = (legende + " " + contexte).lower()
    for indice in tableau_indices:
        if indice in texte_a_verifier:
            return True
    
    return False

def trouver_legende_image(image, texte_markdown):
    """
    Tente de trouver la légende d'une image dans le texte markdown.
    
    Args:
        image: Objet image du résultat OCR
        texte_markdown: Texte markdown contenant potentiellement la légende
        
    Returns:
        Légende trouvée ou chaîne vide
    """
    # Chercher la référence à l'image avec sa légende
    pattern = f"!\\[(.*?)\\]\\({image.id}\\)"
    match = re.search(pattern, texte_markdown)
    
    if match and match.group(1).strip():
        return match.group(1).strip()
    
    # Chercher une légende à proximité
    patterns_legende = [
        r"Fig(?:ure)?\.?\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
        r"Tableau\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
        r"Table\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
        r"Image\s+\d+\s*:?\s*(.*?)\.(?:\n|$)"
    ]
    
    # Trouver où se trouve la référence à l'image
    ref_match = re.search(f"!\\[.*?\\]\\({image.id}\\)", texte_markdown)
    if ref_match:
        # Rechercher dans les 500 caractères après la référence
        texte_apres = texte_markdown[ref_match.end():ref_match.end()+500]
        
        for pattern in patterns_legende:
            match = re.search(pattern, texte_apres)
            if match:
                return match.group(1).strip()
    
    return ""

def extraire_contexte(image, texte_markdown):
    """
    Extrait le texte environnant une image pour fournir du contexte.
    
    Args:
        image: Objet image du résultat OCR
        texte_markdown: Texte markdown contenant l'image
        
    Returns:
        Texte de contexte
    """
    # Trouver où se trouve la référence à l'image
    ref_match = re.search(f"!\\[.*?\\]\\({image.id}\\)", texte_markdown)
    if not ref_match:
        return ""
    
    # Extraire du texte avant et après la référence
    debut = max(0, ref_match.start() - 300)
    fin = min(len(texte_markdown), ref_match.end() + 300)
    
    # Extraire le contexte
    contexte = texte_markdown[debut:ref_match.start()] + texte_markdown[ref_match.end():fin]
    
    # Nettoyer le texte (supprimer les références aux autres images)
    contexte = re.sub(r'!\[.*?\]\(.*?\)', '', contexte)
    
    return contexte.strip()

# -------- POINT D'ENTRÉE PRINCIPAL --------

if __name__ == "__main__":
    # Traiter les arguments en ligne de commande
    if len(sys.argv) > 1:
        # Un chemin de PDF a été fourni en argument
        PDF_A_ANALYSER = sys.argv[1]
        
        # Vérifier les options
        exporter_elements_seulement = "--export-only" in sys.argv
        sans_export = "--no-export" in sys.argv
    else:
        # Utiliser le PDF par défaut
        PDF_A_ANALYSER = "./data/pdf/BIS_Stablecoins versus tokenised deposits.pdf"
        exporter_elements_seulement = False
        sans_export = False
    
    # Vérifier si le fichier existe
    if not os.path.isfile(PDF_A_ANALYSER):
        print(f"ERREUR: Le fichier '{PDF_A_ANALYSER}' n'existe pas.")
        print("Veuillez spécifier un chemin valide pour votre PDF.")
        exit(1)
    
    # Exporter uniquement ou analyser + exporter
    if exporter_elements_seulement:
        # Exporter les éléments sans réanalyser le PDF
        dossier_export = exporter_elements(PDF_A_ANALYSER)
        
        if dossier_export:
            print(f"\nVous pouvez accéder aux éléments exportés dans: {dossier_export}")
            print("Structure du dossier:")
            print(f"- {dossier_export}/texte/ - Chunks de texte")
            print(f"- {dossier_export}/texte_complet.txt - Texte intégral")
            print(f"- {dossier_export}/images/ - Images extraites")
            print(f"- {dossier_export}/tableaux/ - Tableaux extraits")
            print(f"- {dossier_export}/resume.txt - Résumé de l'exportation")
    else:
        # Analyser et indexer le PDF (et exporter si demandé)
        resultat = analyser_pdf(PDF_A_ANALYSER, exporter=not sans_export)
        
        # Afficher un résumé
        print("\n" + "="*50)
        print(f"RÉSUMÉ DE L'INDEXATION:")
        print(f"- Fichier: {resultat['fichier']}")
        print(f"- Chunks de texte: {resultat['chunks_texte']}")
        print(f"- Images: {resultat['images']}")
        print(f"- Tableaux: {resultat['tableaux']}")
        print(f"- Total éléments stockés: {len(resultat['ids_stockes'])}")
        print(f"- Base de données: {DB_PATH}")
        
        # Informations sur l'exportation
        if not sans_export and 'dossier_export' in resultat:
            print(f"- Éléments exportés dans: {resultat['dossier_export']}")
        
        print("="*50)
        
        print("\nPour utiliser ces données dans votre système RAG:")
        print("1. Connectez-vous à la base vectorielle ChromaDB dans le dossier:", DB_PATH)
        print("2. Utilisez la collection 'pdf_collection' pour effectuer vos recherches")
        print("3. Filtrez par type (text, image, table) selon vos besoins")
        print("4. Les images et tableaux sont sauvegardés dans le dossier:", IMAGE_DIR)
        
        if not sans_export and 'dossier_export' in resultat:
            print("\nLes éléments exportés sont disponibles dans:")
            print(f"- {resultat['dossier_export']}/texte/ - Chunks de texte")
            print(f"- {resultat['dossier_export']}/texte_complet.txt - Texte intégral")
            print(f"- {resultat['dossier_export']}/images/ - Images extraites")
            print(f"- {resultat['dossier_export']}/tableaux/ - Tableaux extraits")
            print(f"- {resultat['dossier_export']}/resume.txt - Résumé de l'exportation")
        
        # Demander à l'utilisateur s'il souhaite visualiser les éléments
        choix = input("\nSouhaitez-vous visualiser les éléments exportés? (o/n): ").lower()
        if choix == 'o' or choix == 'oui':
            visualiser_elements_exportes(resultat['fichier'])

def visualiser_elements_exportes(chemin_pdf):
    """
    Visualise les éléments exportés (images, tableaux et chunks de texte).
    
    Args:
        chemin_pdf: Chemin du fichier PDF analysé
    """
    # Initialiser la connexion à ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("pdf_collection")
    
    # Générer l'ID du document
    pdf_id = hashlib.md5(chemin_pdf.encode()).hexdigest()
    
    # Afficher un menu
    print("\n" + "="*50)
    print("VISUALISATION DES ÉLÉMENTS EXPORTÉS")
    print("="*50)
    
    while True:
        print("\nQue souhaitez-vous visualiser?")
        print("1. Chunks de texte")
        print("2. Images")
        print("3. Tableaux")
        print("4. Quitter")
        
        choix = input("Votre choix (1-4): ")
        
        if choix == "1":
            # Récupérer et afficher les chunks de texte
            resultats = collection.get(
                where={"document_id": pdf_id, "type": "text"},
                include=["documents", "metadatas"]
            )
            
            if not resultats["documents"]:
                print("Aucun chunk de texte trouvé.")
                continue
                
            print(f"\n{len(resultats['documents'])} CHUNKS DE TEXTE TROUVÉS:")
            for i, (doc, meta) in enumerate(zip(resultats["documents"], resultats["metadatas"])):
                print(f"\n--- CHUNK {i+1}/{len(resultats['documents'])} ---")
                print(f"Index: {meta.get('chunk_index')}/{meta.get('chunk_count')-1}")
                print("-"*50)
                print(doc[:300] + "..." if len(doc) > 300 else doc)
                
                if i < len(resultats["documents"]) - 1:
                    suite = input("\nAfficher le chunk suivant? (o/n, ou 'q' pour quitter): ").lower()
                    if suite == 'q':
                        break
                    elif suite != 'o' and suite != 'oui':
                        break
        
        elif choix == "2":
            # Récupérer et afficher les informations sur les images
            resultats = collection.get(
                where={"document_id": pdf_id, "type": "image"},
                include=["documents", "metadatas"]
            )
            
            if not resultats["documents"]:
                print("Aucune image trouvée.")
                continue
                
            print(f"\n{len(resultats['documents'])} IMAGES TROUVÉES:")
            for i, (doc, meta) in enumerate(zip(resultats["documents"], resultats["metadatas"])):
                print(f"\n--- IMAGE {i+1}/{len(resultats['documents'])} ---")
                print(f"Page: {meta.get('page')}")
                print(f"Chemin: {meta.get('image_path')}")
                if meta.get('caption'):
                    print(f"Légende: {meta.get('caption')}")
                print(f"Taille: {meta.get('width')}x{meta.get('height')}")
                
                # Proposer d'ouvrir l'image
                if os.path.exists(meta.get('image_path', '')):
                    ouvrir = input("\nOuvrir cette image? (o/n): ").lower()
                    if ouvrir == 'o' or ouvrir == 'oui':
                        try:
                            # Sur macOS
                            os.system(f"open '{meta.get('image_path')}'")
                        except:
                            print(f"Impossible d'ouvrir l'image. Vous pouvez la trouver à: {meta.get('image_path')}")
                
                if i < len(resultats["documents"]) - 1:
                    suite = input("\nAfficher l'image suivante? (o/n, ou 'q' pour quitter): ").lower()
                    if suite == 'q':
                        break
                    elif suite != 'o' and suite != 'oui':
                        break
        
        elif choix == "3":
            # Récupérer et afficher les informations sur les tableaux
            resultats = collection.get(
                where={"document_id": pdf_id, "type": "table"},
                include=["documents", "metadatas"]
            )
            
            if not resultats["documents"]:
                print("Aucun tableau trouvé.")
                continue
                
            print(f"\n{len(resultats['documents'])} TABLEAUX TROUVÉS:")
            for i, (doc, meta) in enumerate(zip(resultats["documents"], resultats["metadatas"])):
                print(f"\n--- TABLEAU {i+1}/{len(resultats['documents'])} ---")
                print(f"Page: {meta.get('page')}")
                print(f"Chemin: {meta.get('table_path')}")
                if meta.get('caption'):
                    print(f"Légende: {meta.get('caption')}")
                print(f"Taille: {meta.get('width')}x{meta.get('height')}")
                
                # Proposer d'ouvrir le tableau
                if os.path.exists(meta.get('table_path', '')):
                    ouvrir = input("\nOuvrir ce tableau? (o/n): ").lower()
                    if ouvrir == 'o' or ouvrir == 'oui':
                        try:
                            # Sur macOS
                            os.system(f"open '{meta.get('table_path')}'")
                        except:
                            print(f"Impossible d'ouvrir le tableau. Vous pouvez le trouver à: {meta.get('table_path')}")
                
                if i < len(resultats["documents"]) - 1:
                    suite = input("\nAfficher le tableau suivant? (o/n, ou 'q' pour quitter): ").lower()
                    if suite == 'q':
                        break
                    elif suite != 'o' and suite != 'oui':
                        break
        
        elif choix == "4":
            break
        
        else:
            print("Choix non valide. Veuillez réessayer.") 