"""
ANALYSEUR PDF AVEC MISTRAL OCR POUR RAG
---------------------------------------
Ce script permet d'extraire le texte, les images et les tableaux d'un document PDF,
de générer des embeddings, de stocker ces données dans une base vectorielle,
et d'exporter les éléments dans un dossier local.
"""

import os
import re
import sys
import shutil
import hashlib
from datetime import datetime
from io import BytesIO

from PIL import Image
import datauri
from mistralai import Mistral

from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------- CONFIGURATION --------
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

IMAGE_DIR = "data/output/images"
TABLE_DIR = "data/output/tables" 
EXPORT_BASE_DIR = "data/output"
for d in [IMAGE_DIR, TABLE_DIR, EXPORT_BASE_DIR]:
    os.makedirs(d, exist_ok=True)

# -------- CONFIGURATION CHUNKING AMÉLIORÉE --------
# Basé sur les meilleures pratiques RAG 2024
CHUNK_SIZE = 1500  # Taille augmentée pour plus de contexte (≈300-400 mots)
CHUNK_OVERLAP = 225  # 15% d'overlap optimal (CHUNK_SIZE * 0.15)

# Nouveaux paramètres pour chunking intelligent
MIN_CHUNK_SIZE = 300  # Éviter les chunks trop petits
MAX_CHUNK_SIZE = 2000  # Limite supérieure stricte

# 🆕 MODÈLE D'EMBEDDING UNIFIÉ - Cohérent avec agent_config.py
from agents.config.agent_config import EMBEDDING_MODEL_NAME
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME  # Utilise le modèle centralisé

print(f"📊 PDF Processor: Using embedding model: {EMBEDDING_MODEL}")

# -------- FONCTION PRINCIPALE --------

def analyser_pdf(chemin_pdf, db_path: str, exporter=True):
    """Analyse un PDF, extrait les éléments et les stocke dans une base vectorielle.

    Args:
        chemin_pdf (str): Chemin vers le fichier PDF.
        db_path (str): Chemin vers le dossier de la base de données vectorielle ChromaDB.
        exporter (bool): Si True, exporte les éléments extraits localement.

    Returns:
        dict: Résumé de l'analyse.
    """
    print(f"Analyse du PDF: {chemin_pdf}")
    print(f"Utilisation de la base de données: {db_path}")
    client = Mistral(api_key=MISTRAL_API_KEY)

    # 1. Traitement OCR
    resultat_ocr = traiter_pdf_avec_mistral(client, chemin_pdf)

    # 2. Extraction des contenus
    texte = extraire_texte_propre(resultat_ocr)
    images = extraire_images_sans_tableaux(resultat_ocr)
    tableaux = detecter_tableaux(resultat_ocr)
    metadonnees = extraire_metadonnees(resultat_ocr, chemin_pdf)

    # 3. Découpage du texte avec tracking des pages
    chunks_avec_pages = decouper_texte_avec_pages(resultat_ocr)
    chunks = [chunk_info['content'] for chunk_info in chunks_avec_pages]

    # 4. Calcul des embeddings
    modele_embeddings = SentenceTransformer(EMBEDDING_MODEL)
    emb_text = generer_embeddings(modele_embeddings, chunks)
    emb_imgs = generer_embeddings_images(modele_embeddings, images)
    emb_tabs = generer_embeddings_images(modele_embeddings, tableaux, est_tableau=True)

    # 5. Stockage vectoriel
    stock_ids = stocker_dans_base_vectorielle(
        db_path,
        chemin_pdf,
        chunks, emb_text, chunks_avec_pages,
        images, emb_imgs,
        tableaux, emb_tabs,
        metadonnees
    )

    resume = {
        "fichier": chemin_pdf,
        "chunks_texte": len(chunks),
        "images": len(images),
        "tableaux": len(tableaux),
        "ids_stockes": stock_ids
    }
    print(f"Analyse terminée : {len(chunks)} chunks, {len(images)} images, {len(tableaux)} tableaux")

    # 6. Export des contenus et suppression des fichiers temporaires
    if exporter:
        dossier_export = exporter_elements(chemin_pdf, texte, chunks)
        resume["dossier_export"] = dossier_export

    return resume


# -------- TRAITEMENT OCR --------

def traiter_pdf_avec_mistral(client, chemin_pdf):
    print("Envoi du PDF à Mistral OCR...")
    with open(chemin_pdf, "rb") as f:
        pdf_data = f.read()
    uploaded = client.files.upload(
        file={"file_name": chemin_pdf, "content": pdf_data},
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        include_image_base64=True
    )
    print("Traitement OCR terminé")
    return response


# -------- EXTRACTION DU TEXTE --------

def extraire_texte_propre(resultat_ocr):
    print("Extraction du texte...")
    texte_complet = ""
    for page in resultat_ocr.pages:
        if page.markdown:
            texte_complet += page.markdown + "\n\n"
    return texte_complet


# -------- EXTRACTION DES IMAGES --------

def extraire_images_sans_tableaux(resultat_ocr):
    """
    Extrait les images en ignorant celles dont l'ID est présent dans la propriété 'tables' de la page.
    Enregistre les images sous "pageX_imageY.jpeg" dans IMAGE_DIR.
    """
    print("Extraction des images (en excluant les tableaux) ...")
    images_info = []
    for page_idx, page in enumerate(resultat_ocr.pages, 1):
        table_ids = set()
        if hasattr(page, "tables") and page.tables:
            for t in page.tables:
                if hasattr(t, "id"):
                    table_ids.add(t.id)
        if not page.images:
            continue
        for i, image in enumerate(page.images):
            if hasattr(image, "id") and image.id in table_ids:
                continue  # Ignorer l'image déjà identifiée comme tableau
            try:
                data = datauri.parse(image.image_base64).data
                pil_img = Image.open(BytesIO(data))
                image_name = f"page{page_idx}_image{i}.jpeg"
                image_path = os.path.join(IMAGE_DIR, image_name)
                pil_img.save(image_path)
                legende = extraire_legende(image, page.markdown)
                contexte = extraire_contexte(image, page.markdown)
                images_info.append({
                    "page": page_idx,
                    "chemin": image_path,
                    "id": image.id,
                    "largeur": pil_img.width,
                    "hauteur": pil_img.height,
                    "legende": legende,
                    "type": "image",
                    "contexte": contexte
                })
                print(f"  Image extraite: {image_path}")
            except Exception as e:
                print(f"Erreur lors du traitement d'une image: {str(e)}")
    print(f"  Total images extraites: {len(images_info)}")
    return images_info


# -------- EXTRACTION DES TABLEAUX --------

def detecter_tableaux(resultat_ocr):
    """
    Extrait les tableaux depuis la propriété 'tables' si disponible,
    sinon utilise un fallback par regex.
    Les tableaux sont enregistrés dans TABLE_DIR uniquement.
    Un contrôle vérifie que le bloc comporte bien un séparateur Markdown.
    """
    print("Extraction des tableaux...")
    tableaux_info = []

    # Méthode 1 : Utilisation de la propriété 'tables'
    for page_idx, page in enumerate(resultat_ocr.pages, 1):
        if hasattr(page, "tables") and page.tables:
            for idx, tbl in enumerate(page.tables):
                tbl_markdown = getattr(tbl, "markdown", "")
                cleaned, _ = clean_table_structure(tbl_markdown)
                if not cleaned.strip():
                    continue
                table_name = f"page{page_idx}_tableau_{idx}.txt"
                table_path = os.path.join(TABLE_DIR, table_name)
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                tableaux_info.append({
                    "page": page_idx,
                    "chemin": table_path,
                    "id": getattr(tbl, "id", f"table_{page_idx}_{idx}"),
                    "legende": "",
                    "type": "tableau",
                    "contexte": cleaned,
                    "description": f"Tableau extrait à la page {page_idx} (via propriété)",
                    "largeur": 0,
                    "hauteur": 0
                })
                print(f"  Tableau extrait sur page {page_idx} -> {table_path}")

    # Méthode 2 : Fallback par regex
    if not tableaux_info:
        regex = re.compile(r"""
            (
              ^\|.*\|\s*\n
              ^\|(?:\s*[:-]+\s*\|)+\s*\n
              (?:^\|.*\|\s*\n)*
            )
        """, re.MULTILINE | re.VERBOSE)
        for page_idx, page in enumerate(resultat_ocr.pages, 1):
            if page.markdown:
                found = regex.findall(page.markdown)
                for idx, table_text in enumerate(found):
                    cleaned, _ = clean_table_structure(table_text)
                    if not re.search(r"^\|\s*[-:]+\s*\|", cleaned, re.MULTILINE):
                        continue
                    table_name = f"page{page_idx}_tableau_{idx}.txt"
                    table_path = os.path.join(TABLE_DIR, table_name)
                    with open(table_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    tableaux_info.append({
                        "page": page_idx,
                        "chemin": table_path,
                        "id": f"table_{page_idx}_{idx}",
                        "legende": "",
                        "type": "tableau",
                        "contexte": cleaned,
                        "description": f"Tableau extrait à la page {page_idx} (via regex)",
                        "largeur": 0,
                        "hauteur": 0
                    })
                    print(f"  (Fallback) Tableau détecté sur page {page_idx} -> {table_path}")

    print(f"  {len(tableaux_info)} tableaux extraits au total")
    return tableaux_info

def clean_table_structure(table_text):
    lines = table_text.strip().split("\n")
    rows = [ [cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines ]
    if not rows:
        return table_text, 0
    max_cols = max(len(row) for row in rows)
    corrected_lines = []
    modif_count = 0
    for original, row in zip(lines, rows):
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
            modif_count += 1
        corrected_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(corrected_lines), modif_count


# -------- MÉTADONNÉES --------

def extraire_metadonnees(resultat_ocr, chemin_pdf):
    print("Extraction des métadonnées...")
    meta = {
        "nom_fichier": os.path.basename(chemin_pdf),
        "chemin": chemin_pdf,
        "nombre_pages": len(resultat_ocr.pages)
    }
    debut = ""
    for i, page in enumerate(resultat_ocr.pages):
        if i < 2 and page.markdown:
            debut += page.markdown + "\n"
    for pat in [r"# (.*?)[\n\r]", r"^(.{5,100})[\n\r]"]:
        m = re.search(pat, debut, re.MULTILINE)
        if m:
            meta["titre"] = m.group(1).strip()
            break
    for pat in [r"(?:Author|Auteur)[\s:]+(.{3,100})", r"(?:By|Par)[\s:]+(.{3,100})"]:
        m = re.search(pat, debut, re.MULTILINE | re.IGNORECASE)
        if m:
            meta["auteur"] = m.group(1).strip()
            break
    for pat in [r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\b"]:
        m = re.search(pat, debut, re.MULTILINE)
        if m:
            meta["date"] = m.group(1).strip()
            break
    return meta


# -------- DÉCOUPAGE DU TEXTE --------

def decouper_texte(texte):
    print(f"Découpage du texte (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(texte)
    print(f"  Texte découpé en {len(chunks)} chunks")
    return chunks


def decouper_texte_avec_pages(resultat_ocr):
    """
    Découpe le texte en chunks en gardant l'information de la page d'origine.
    VERSION AMÉLIORÉE avec chunking sémantique intelligent.
    
    Args:
        resultat_ocr: Résultat de l'OCR avec les pages
        
    Returns:
        List[dict]: Liste de dictionnaires avec 'content' et 'page_info'
    """
    print(f"🧠 Découpage intelligent du texte (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
    
    # Stratégie de séparation améliorée pour préserver la structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",  # Sections importantes
            "\n\n",    # Paragraphes
            "\n",      # Lignes
            ". ",      # Phrases
            "! ",      # Exclamations
            "? ",      # Questions
            "; ",      # Points-virgules
            ", ",      # Virgules
            " ",       # Espaces
            ""         # Caractères
        ],
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks_avec_pages = []
    
    for page_idx, page in enumerate(resultat_ocr.pages, 1):
        if not page.markdown:
            continue
            
        # ✨ Nettoyage intelligent du texte
        page_text = _preprocess_text_for_rag(page.markdown)
        
        if not page_text or len(page_text.strip()) < MIN_CHUNK_SIZE:
            continue
            
        # Découper le texte de cette page en chunks
        page_chunks = splitter.split_text(page_text)
        
        # ✨ Filtrage et enrichissement des chunks
        for chunk_idx, chunk in enumerate(page_chunks):
            # Éviter les chunks trop petits ou vides
            if len(chunk.strip()) < MIN_CHUNK_SIZE:
                continue
                
            # Limiter la taille maximale
            if len(chunk) > MAX_CHUNK_SIZE:
                chunk = chunk[:MAX_CHUNK_SIZE] + "..."
            
            # Ajouter un contexte de page si nécessaire
            enriched_chunk = _add_page_context(chunk, page_idx, page_text)
            
            chunks_avec_pages.append({
                'content': enriched_chunk,
                'page_number': page_idx,
                'chunk_in_page': chunk_idx,
                'total_chunks_in_page': len(page_chunks),
                'chunk_length': len(enriched_chunk),
                'original_length': len(chunk)
            })
    
    print(f"✅ Texte découpé en {len(chunks_avec_pages)} chunks intelligents avec information de page")
    return chunks_avec_pages


def _preprocess_text_for_rag(text: str) -> str:
    """
    Préprocessing intelligent du texte pour améliorer la qualité RAG.
    """
    # Supprimer les références d'images mais garder les légendes utiles
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'Image: \1', text)
    
    # Normaliser les sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r' {2,}', ' ', text)
    
    # Nettoyer les caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Préserver la structure des listes et énumérations
    text = re.sub(r'(\d+\.)\s+', r'\1 ', text)  # Normaliser les listes numérotées
    text = re.sub(r'([•\-\*])\s+', r'\1 ', text)  # Normaliser les puces
    
    return text.strip()


def _add_page_context(chunk: str, page_number: int, full_page_text: str) -> str:
    """
    Ajoute un contexte de page minimal pour améliorer la recherche.
    """
    # Ajouter le numéro de page de manière discrète
    context_prefix = f"[Page {page_number}] "
    
    # Si le chunk est court, ajouter un peu plus de contexte
    if len(chunk) < CHUNK_SIZE * 0.7:  # Si le chunk fait moins de 70% de la taille cible
        # Essayer de trouver un titre ou en-tête dans la page
        lines = full_page_text.split('\n')
        for line in lines[:5]:  # Regarder les 5 premières lignes
            if line.strip() and (line.isupper() or line.startswith('#')):
                context_prefix += f"{line.strip()}: "
                break
    
    return context_prefix + chunk


# -------- EMBEDDINGS --------

def generer_embeddings(modele, textes):
    print(f"Génération d'embeddings pour {len(textes)} textes...")
    return modele.encode(textes, show_progress_bar=False)

def generer_embeddings_images(modele, elements, est_tableau=False):
    type_label = "tableaux" if est_tableau else "images"
    print(f"Génération d'embeddings pour {len(elements)} {type_label}...")
    descriptions = []
    for el in elements:
        desc = f"{'Tableau' if est_tableau else 'Image'} à la page {el['page']}"
        if el.get("legende"):
            desc += f". Légende: {el['legende']}"
        if el.get("contexte"):
            desc += f". Contexte: {el['contexte'][:200]}"
        descriptions.append(desc)
    if not descriptions:
        return []
    return modele.encode(descriptions, show_progress_bar=False)


# -------- STOCKAGE VECTORIEL --------

def stocker_dans_base_vectorielle(db_path: str, chemin_pdf, chunks, emb_textes, chunks_avec_pages, images, emb_images, tableaux, emb_tableaux, metadonnees):
    print("Stockage dans la base vectorielle...")
    # Ensure the directory exists before creating the client
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "pdf_collection"
    try:
        collection = client.get_collection(collection_name)
        print(f"  Collection '{collection_name}' trouvée")
    except Exception:
        collection = client.create_collection(collection_name)
        print(f"  Collection '{collection_name}' créée")

    pdf_id = hashlib.md5(chemin_pdf.encode()).hexdigest()
    nom_fichier = os.path.basename(chemin_pdf)
    all_ids = []

    # 1) Indexation du texte avec informations de page
    for i, (chunk, emb, chunk_info) in enumerate(zip(chunks, emb_textes, chunks_avec_pages)):
        doc_id = f"{pdf_id}_txt_{i}"
        all_ids.append(doc_id)
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "source": nom_fichier,
            "type": "text",
            "chunk_index": i,
            "chunk_count": len(chunks),
            "page": chunk_info['page_number'],
            "chunk_in_page": chunk_info['chunk_in_page'],
            "total_chunks_in_page": chunk_info['total_chunks_in_page']
        }
        meta.update({f"doc_{k}": v for k, v in metadonnees.items()})
        collection.add(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            metadatas=[meta],
            documents=[chunk]
        )

    # 2) Indexation des images
    for i, (img, emb) in enumerate(zip(images, emb_images)):
        doc_id = f"{pdf_id}_img_{i}"
        all_ids.append(doc_id)
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "source": nom_fichier,
            "type": "image",
            "page": img["page"],
            "image_path": img["chemin"],
            "width": img["largeur"],
            "height": img["hauteur"]
        }
        if img.get("legende"):
            meta["caption"] = img["legende"]
        desc = f"Image à la page {img['page']}"
        if img.get("legende"):
            desc += f": {img['legende']}"
        collection.add(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            metadatas=[meta],
            documents=[desc]
        )

    # 3) Indexation des tableaux
    for i, (tab, emb) in enumerate(zip(tableaux, emb_tableaux)):
        doc_id = f"{pdf_id}_tab_{i}"
        all_ids.append(doc_id)
        meta = {
            "document_id": pdf_id,
            "filename": nom_fichier,
            "source": nom_fichier,
            "type": "table",
            "page": tab["page"],
            "table_path": tab["chemin"],
            "width": tab["largeur"],
            "height": tab["hauteur"]
        }
        if tab.get("legende"):
            meta["caption"] = tab["legende"]
        desc = tab.get("description", f"Tableau à la page {tab['page']}")
        collection.add(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            metadatas=[meta],
            documents=[desc]
        )

    print(f"  {len(all_ids)} éléments stockés dans la base vectorielle")
    return all_ids


# -------- FONCTIONS AUXILIAIRES --------

def extraire_legende(image, texte_markdown):
    pattern = f"!\\[(.*?)\\]\\({image.id}\\)"
    m = re.search(pattern, texte_markdown)
    return m.group(1).strip() if m and m.group(1).strip() else ""

def extraire_contexte(image, texte_markdown, contexte_taille=300):
    m = re.search(f"!\\[.*?\\]\\({image.id}\\)", texte_markdown)
    if not m:
        return ""
    debut = max(0, m.start() - contexte_taille)
    fin = min(len(texte_markdown), m.end() + contexte_taille)
    contexte = texte_markdown[debut:m.start()] + texte_markdown[m.end():fin]
    return re.sub(r'!\[.*?\]\(.*?\)', '', contexte).strip()


def vider_dossier(dir_path):
    """
    Supprime tous les fichiers dans un dossier sans supprimer le dossier lui-même.
    """
    for fichier in os.listdir(dir_path):
        chemin = os.path.join(dir_path, fichier)
        if os.path.isfile(chemin):
            os.remove(chemin)
        else:
            shutil.rmtree(chemin)


def exporter_elements(chemin_pdf, texte_complet, chunks):
    """
    Exporte le texte complet et les chunks dans /texte,
    copie les images depuis IMAGE_DIR vers /images,
    copie les tableaux depuis TABLE_DIR vers /tableaux.
    Ensuite, vide les dossiers IMAGE_DIR et TABLE_DIR pour éviter les doublons.
    """
    nom_pdf = os.path.basename(chemin_pdf).replace('.pdf', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(EXPORT_BASE_DIR, f"{nom_pdf}_{timestamp}")

    for sub in ["texte", "images", "tableaux"]:
        os.makedirs(os.path.join(export_dir, sub), exist_ok=True)

    print(f"Exportation des éléments pour {chemin_pdf} vers {export_dir} ...")
    try:
        # 1) Texte complet
        chemin_tc = os.path.join(export_dir, "texte", "texte_complet.txt")
        with open(chemin_tc, "w", encoding="utf-8") as f:
            f.write(texte_complet)

        # 2) Chunks de texte
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(export_dir, "texte", f"chunk_{i:03d}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)

        # 3) Copier les images depuis IMAGE_DIR
        for file in os.listdir(IMAGE_DIR):
            src = os.path.join(IMAGE_DIR, file)
            dst = os.path.join(export_dir, "images", file)
            shutil.copy2(src, dst)

        # 4) Copier les tableaux depuis TABLE_DIR
        for file in os.listdir(TABLE_DIR):
            src = os.path.join(TABLE_DIR, file)
            dst = os.path.join(export_dir, "tableaux", file)
            shutil.copy2(src, dst)

        print(f"Exportation terminée dans: {export_dir}")

        # Suppression des fichiers temporaires
        vider_dossier(IMAGE_DIR)
        vider_dossier(TABLE_DIR)

        return export_dir
    except Exception as e:
        print(f"Erreur lors de l'exportation: {str(e)}")
        return None


# -------- POINT D'ENTRÉE --------

if __name__ == "__main__":
    # Choix du PDF par argument ou par input
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Veuillez entrer le chemin du fichier PDF à analyser: ")

    pdf_path = os.path.abspath(pdf_path)
    print(f"Chemin absolu: {pdf_path}")

    if not os.path.isfile(pdf_path):
        print(f"ERREUR: Le fichier '{pdf_path}' n'existe pas.")
        sys.exit(1)

    # Instead of asking for theme, let's use a fixed path for direct script execution
    # Or you could keep the theme logic if needed for standalone runs
    script_db_path = os.path.join("data", "output", "vectordb", "standalone_run")
    print(f"Base de données pour exécution directe: {script_db_path}")

    resultat = analyser_pdf(pdf_path, db_path=script_db_path, exporter=True)

    print("\n" + "=" * 50)
    print("RÉSUMÉ DE L'INDEXATION:")
    print(f"- Fichier: {resultat['fichier']}")
    print(f"- Chunks de texte: {resultat['chunks_texte']}")
    print(f"- Images: {resultat['images']}")
    print(f"- Tableaux: {resultat['tableaux']}")
    print(f"- Total éléments stockés: {len(resultat['ids_stockes'])}")
    print(f"- Base de données: {script_db_path}")

    if "dossier_export" in resultat:
        print(f"\nÉléments exportés dans: {resultat['dossier_export']}")
        print("  |- texte/ (texte_complet.txt + chunks)")
        print("  |- images/")
        print("  |- tableaux/")