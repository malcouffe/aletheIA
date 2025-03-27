#!/usr/bin/env python3
"""
Script pour ajouter tous les fichiers PDF d'un répertoire à la base de données Chroma.
"""
import os
import sys
import argparse
from utils.pipeline_indexation.pdf import PDFProcessor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ajouter un répertoire de fichiers PDF à la base de données Chroma")
    parser.add_argument("directory_path", help="Chemin vers le répertoire contenant les fichiers PDF")
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
    
    # Vérifier si le répertoire existe
    if not os.path.isdir(args.directory_path):
        print(f"Erreur: Le répertoire '{args.directory_path}' n'existe pas.")
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
    
    print(f"Traitement des fichiers PDF dans: {args.directory_path}")
    if args.use_mistral_ocr:
        print("Utilisation de Mistral OCR pour l'extraction de texte et d'images")
    
    try:
        results = processor.process_directory(args.directory_path)
        
        # Résumé des résultats
        total_files = len(results)
        total_chunks = sum(len(chunks) for chunks in results.values())
        
        # Compter combien d'IDs sont des images vs du texte
        total_text_chunks = 0
        total_images = 0
        for file_chunks in results.values():
            text_chunks = [id for id in file_chunks if "_txt_" in id]
            image_chunks = [id for id in file_chunks if "_img_" in id]
            total_text_chunks += len(text_chunks)
            total_images += len(image_chunks)
        
        print(f"\nRésumé:")
        print(f"- Fichiers PDF traités: {total_files}")
        print(f"- Total d'éléments ajoutés: {total_chunks}")
        print(f"  - Chunks de texte: {total_text_chunks}")
        print(f"  - Images: {total_images}")
        print(f"- Base de données: {args.db_path}")
        print(f"- Collection: {args.collection}")
        
        # Détails par fichier
        print("\nDétails par fichier:")
        for filename, chunks in results.items():
            text_chunks = [id for id in chunks if "_txt_" in id]
            image_chunks = [id for id in chunks if "_img_" in id]
            print(f"- {filename}: {len(chunks)} éléments ({len(text_chunks)} textes, {len(image_chunks)} images)")
            
    except Exception as e:
        print(f"Erreur lors du traitement du répertoire: {str(e)}")
        sys.exit(1)
    
    print("\nPour chercher dans vos documents indexés, utilisez le script search.py:")
    print(f"python search.py \"votre requête de recherche\" --db-path {args.db_path} --collection {args.collection}")
    print("\nPour chercher spécifiquement les images:")
    print(f"python search.py \"votre requête de recherche\" --db-path {args.db_path} --collection {args.collection} --filter content_type=image")

if __name__ == "__main__":
    main() 