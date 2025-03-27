#!/usr/bin/env python3
"""
Script pour ajouter un fichier PDF à la base de données Chroma.
Ce script peut être exécuté à chaque fois que vous souhaitez ajouter un nouveau PDF.
"""
import os
import sys
import argparse
from utils.pipeline_indexation.pdf import PDFProcessor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ajouter un fichier PDF à la base de données Chroma")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF à indexer")
    parser.add_argument("--db-path", default="./pdf_database", help="Chemin vers la base de données Chroma (défaut: ./pdf_database)")
    parser.add_argument("--collection", default="pdf_documents", help="Nom de la collection dans Chroma (défaut: pdf_documents)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Taille des chunks de texte (défaut: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chevauchement entre les chunks (défaut: 200)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L12-v2", help="Modèle HuggingFace pour les embeddings")
    parser.add_argument("--no-images", action="store_true", help="Ne pas extraire les images du PDF")
    parser.add_argument("--image-min-size", type=int, default=100, help="Taille minimale des images à extraire en pixels (défaut: 100)")
    parser.add_argument("--store-image-content", action="store_true", help="Stocker le contenu des images en base64 dans les métadonnées")
    parser.add_argument("--use-mistral-ocr", action="store_true", help="Utiliser Mistral OCR pour le traitement du PDF")
    parser.add_argument("--mistral-api-key", help="Clé API pour Mistral AI (requise si --use-mistral-ocr est utilisé)")
    parser.add_argument("--image-output-dir", default="./images_extraites", help="Répertoire pour sauvegarder les images extraites (défaut: ./images_extraites)")
    
    args = parser.parse_args()
    
    # Vérifier si le fichier PDF existe
    if not os.path.isfile(args.pdf_path):
        print(f"Erreur: Le fichier '{args.pdf_path}' n'existe pas.")
        sys.exit(1)
    
    if not args.pdf_path.lower().endswith('.pdf'):
        print(f"Erreur: Le fichier '{args.pdf_path}' n'est pas un fichier PDF.")
        sys.exit(1)
    
    # Vérifier si l'API key est fournie lorsque Mistral OCR est activé
    if args.use_mistral_ocr and not args.mistral_api_key:
        # Essayer de récupérer la clé API depuis les variables d'environnement
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            print("Erreur: L'option --mistral-api-key est requise lorsque --use-mistral-ocr est activé.")
            print("Vous pouvez également définir la variable d'environnement MISTRAL_API_KEY.")
            sys.exit(1)
        args.mistral_api_key = mistral_api_key
    
    # Créer le répertoire de sortie pour les images si nécessaire
    if args.use_mistral_ocr and not os.path.exists(args.image_output_dir):
        os.makedirs(args.image_output_dir, exist_ok=True)
        print(f"Répertoire créé pour les images extraites: {args.image_output_dir}")
    
    # Initialize the PDF processor
    processor = PDFProcessor(
        chroma_path=args.db_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        extract_images=not args.no_images,
        image_min_size=args.image_min_size,
        store_image_content=args.store_image_content,
        use_mistral_ocr=args.use_mistral_ocr,
        mistral_api_key=args.mistral_api_key,
        image_output_dir=args.image_output_dir
    )
    
    print(f"Traitement du fichier PDF: {args.pdf_path}")
    if args.use_mistral_ocr:
        print("Utilisation de Mistral OCR pour l'extraction de texte et d'images")
    
    try:
        doc_ids = processor.process_pdf(args.pdf_path)
        
        # Compter combien d'IDs sont des images vs du texte
        image_ids = [id for id in doc_ids if "_img_" in id]
        text_ids = [id for id in doc_ids if "_txt_" in id]
        
        print(f"Succès! {len(doc_ids)} éléments ont été ajoutés à la base de données:")
        print(f"- {len(text_ids)} chunks de texte")
        print(f"- {len(image_ids)} images")
        print(f"Base de données: {args.db_path}")
        print(f"Collection: {args.collection}")
    except Exception as e:
        print(f"Erreur lors du traitement du fichier: {str(e)}")
        sys.exit(1)
    
    print("\nPour chercher dans vos documents indexés, utilisez le script search.py:")
    print(f"python search.py \"votre requête de recherche\" --db-path {args.db_path} --collection {args.collection}")
    print("\nPour chercher spécifiquement les images:")
    print(f"python search.py \"votre requête de recherche\" --db-path {args.db_path} --collection {args.collection} --filter content_type=image")

if __name__ == "__main__":
    main() 