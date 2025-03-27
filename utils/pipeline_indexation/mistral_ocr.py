"""
Module pour l'extraction et l'analyse de documents PDF utilisant Mistral OCR.
"""
import os
import re
import base64
import hashlib
from io import BytesIO
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from PIL import Image
import datauri
from mistralai import Mistral
from mistralai.models import OCRResponse

class MistralOCRProcessor:
    """
    Classe pour traiter les documents PDF avec Mistral OCR.
    """
    def __init__(
        self, 
        api_key: str,
        image_output_dir: str = "./images_extraites",
        image_min_size: int = 100
    ):
        """
        Initialiser le processeur Mistral OCR.
        
        Args:
            api_key: Clé API pour Mistral AI
            image_output_dir: Répertoire pour sauvegarder les images extraites
            image_min_size: Taille minimale des images à traiter (en pixels)
        """
        self.api_key = api_key
        self.image_output_dir = image_output_dir
        self.image_min_size = image_min_size
        self.client = self._initialize_client()
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir, exist_ok=True)
    
    def _initialize_client(self) -> Mistral:
        """
        Initialiser le client Mistral AI.
        
        Returns:
            Client Mistral configuré avec la clé API
        """
        return Mistral(api_key=self.api_key)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Traiter un document PDF avec Mistral OCR.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire contenant le texte, les métadonnées et les images extraites
        """
        # Vérifier que le fichier existe
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
        
        # Créer un sous-répertoire spécifique pour ce PDF
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_image_dir = os.path.join(self.image_output_dir, pdf_basename)
        if not os.path.exists(pdf_image_dir):
            os.makedirs(pdf_image_dir, exist_ok=True)
        
        # Effectuer l'OCR
        ocr_response = self._perform_ocr(pdf_path)
        
        # Extraire le texte et les métadonnées
        text_and_metadata = self._extract_text_and_metadata(ocr_response, pdf_path)
        
        # Extraire les images
        images = self._extract_images(ocr_response, pdf_image_dir)
        
        # Combiner les résultats
        result = {
            "text": text_and_metadata["text"],
            "metadata": text_and_metadata["metadata"],
            "images": images
        }
        
        return result
    
    def _perform_ocr(self, pdf_path: str) -> OCRResponse:
        """
        Effectue l'OCR sur le fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Réponse OCR de Mistral
        """
        try:
            # Uploader le fichier
            uploaded_file = self.client.files.upload(
                file={"file_name": pdf_path, "content": open(pdf_path, "rb")},
                purpose="ocr"
            )
            
            # Obtenir l'URL signée
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Effectuer l'OCR
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url},
                include_image_base64=True
            )
            
            return ocr_response
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'OCR avec Mistral: {str(e)}")
    
    def _extract_text_and_metadata(self, ocr_response: OCRResponse, pdf_path: str) -> Dict[str, Any]:
        """
        Extraire le texte et les métadonnées de la réponse OCR.
        
        Args:
            ocr_response: Réponse OCR de Mistral
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire contenant le texte et les métadonnées
        """
        # Extraire le texte de chaque page
        full_text = ""
        for page in ocr_response.pages:
            if page.markdown:
                # Supprimer les références aux images dans le texte pour ne garder que le contenu
                text = re.sub(r'!\[.*?\]\(.*?\)', '', page.markdown)
                full_text += text + "\n\n"
        
        # Extraire les métadonnées de base
        metadata = {
            "filename": os.path.basename(pdf_path),
            "path": pdf_path,
            "page_count": len(ocr_response.pages),
        }
        
        # Analyser le texte pour extraire des métadonnées supplémentaires
        # Titre - probablement le premier texte en gras ou de grande taille
        title_match = re.search(r'# (.*?)\n', full_text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            # Essayer de trouver le premier paragraphe comme titre
            lines = full_text.strip().split('\n')
            if lines:
                metadata["title"] = lines[0].strip()
        
        # Auteur - chercher des patterns typiques
        author_patterns = [
            r"[Aa]uteur\s*:?\s*(.*?)[\n\.]",
            r"[Aa]uthor\s*:?\s*(.*?)[\n\.]",
            r"[Pp]ar\s+(.*?)[\n\.]",
            r"[Bb]y\s+(.*?)[\n\.]"
        ]
        
        for pattern in author_patterns:
            author_match = re.search(pattern, full_text)
            if author_match:
                metadata["author"] = author_match.group(1).strip()
                break
        
        # Mots-clés - chercher des patterns typiques ou faire une extraction basique
        keyword_match = re.search(r"[Mm]ots?[\s-][Cc]lés?:?\s*(.*?)[\n\.]", full_text)
        if keyword_match:
            metadata["keywords"] = keyword_match.group(1).strip()
        else:
            # Extraction simple de mots fréquents
            words = re.findall(r'\b[A-Za-z]{4,}\b', full_text.lower())
            word_freq = {}
            for word in words:
                if word not in ["this", "that", "with", "from", "have", "were", "they", "their"]:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Obtenir les 5 mots les plus fréquents comme mots-clés
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            metadata["extracted_keywords"] = ", ".join([k for k, v in keywords])
        
        return {
            "text": full_text,
            "metadata": metadata
        }
    
    def _extract_images(self, ocr_response: OCRResponse, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extraire et traiter les images de la réponse OCR.
        
        Args:
            ocr_response: Réponse OCR de Mistral
            output_dir: Répertoire de sortie pour les images
            
        Returns:
            Liste de dictionnaires contenant les informations sur les images
        """
        images = []
        image_counter = 0
        
        # Extraire les images de chaque page
        for page_num, page in enumerate(ocr_response.pages, 1):
            if not page.images:
                continue
            
            for img in page.images:
                try:
                    # Extraire l'image depuis base64
                    parsed = datauri.parse(img.image_base64)
                    img_bytes = parsed.data
                    
                    # Créer un objet PIL pour analyser l'image
                    pil_image = Image.open(BytesIO(img_bytes))
                    
                    # Vérifier la taille minimale
                    if pil_image.width < self.image_min_size or pil_image.height < self.image_min_size:
                        continue
                    
                    # Générer un nom unique pour le fichier image
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    img_filename = f"page{page_num}_{img_hash}.jpeg"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Sauvegarder l'image
                    pil_image.save(img_path)
                    
                    # Déterminer le type d'image (figure, graphique, etc.)
                    img_type = self._classify_image(img, page.markdown)
                    
                    # Chercher une légende/contexte pour l'image
                    caption = self._extract_caption_for_image(img, page.markdown)
                    
                    # Créer les informations sur l'image
                    image_info = {
                        "id": f"img_{img_hash}",
                        "page_num": page_num,
                        "img_index": image_counter,
                        "format": "jpeg",
                        "width": pil_image.width,
                        "height": pil_image.height,
                        "type": img_type,
                        "path": img_path,
                        "caption": caption,
                        "context_text": self._extract_context_text(img, page.markdown)
                    }
                    
                    # Ajouter le contenu de l'image en base64 si demandé
                    images.append(image_info)
                    image_counter += 1
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image: {str(e)}")
                    continue
        
        return images
    
    def _classify_image(self, img, markdown_text: str) -> str:
        """
        Classifier le type d'image (graphique, photo, etc.) basé sur le contexte.
        
        Args:
            img: Objet image de la réponse OCR
            markdown_text: Texte markdown de la page
            
        Returns:
            Type d'image ('graph', 'figure', 'image', etc.)
        """
        # Pattern pour trouver la référence à l'image
        img_pattern = f"!\\[.*?\\]\\({img.id}\\)"
        img_match = re.search(img_pattern, markdown_text)
        
        # Texte autour de l'image
        context = markdown_text
        if img_match:
            start_idx = max(0, img_match.start() - 200)
            end_idx = min(len(markdown_text), img_match.end() + 200)
            context = markdown_text[start_idx:end_idx]
        
        # Mots clés pour les différents types d'images
        graph_keywords = ["figure", "fig.", "chart", "graph", "plot", "diagram", "graphique", "tableau"]
        
        # Vérifier si le contexte contient des mots clés pour les graphiques
        for keyword in graph_keywords:
            if keyword.lower() in context.lower():
                if any(k in context.lower() for k in ["chart", "graph", "plot", "graphique"]):
                    return "graph"
                return "figure"
        
        # Par défaut, c'est juste une image
        return "image"
    
    def _extract_caption_for_image(self, img, markdown_text: str) -> str:
        """
        Extraire la légende d'une image basé sur le texte markdown.
        
        Args:
            img: Objet image de la réponse OCR
            markdown_text: Texte markdown de la page
            
        Returns:
            Légende de l'image ou chaîne vide si non trouvée
        """
        # Pattern pour trouver la référence à l'image avec sa légende
        img_pattern = f"!\\[(.*?)\\]\\({img.id}\\)"
        img_match = re.search(img_pattern, markdown_text)
        
        if img_match:
            caption = img_match.group(1).strip()
            if caption:
                return caption
            
            # Si pas de légende dans la balise, chercher après l'image
            after_img = markdown_text[img_match.end():]
            lines = after_img.split('\n')
            if lines and lines[0].strip().startswith('*') and lines[0].strip().endswith('*'):
                return lines[0].strip('* \t')
        
        # Pattern pour trouver du texte qui ressemble à une légende
        caption_patterns = [
            r"Fig(?:ure)?\.?\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
            r"Tableau\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
            r"Table\s+\d+\s*:?\s*(.*?)\.(?:\n|$)",
        ]
        
        for pattern in caption_patterns:
            caption_match = re.search(pattern, markdown_text)
            if caption_match:
                return caption_match.group(1).strip()
        
        return ""
    
    def _extract_context_text(self, img, markdown_text: str) -> str:
        """
        Extraire le texte contextuel autour d'une image.
        
        Args:
            img: Objet image de la réponse OCR
            markdown_text: Texte markdown de la page
            
        Returns:
            Texte contextuel autour de l'image
        """
        # Pattern pour trouver la référence à l'image
        img_pattern = f"!\\[.*?\\]\\({img.id}\\)"
        img_match = re.search(img_pattern, markdown_text)
        
        if not img_match:
            # Si l'image n'est pas trouvée dans le texte, retourner une portion du texte
            return markdown_text[:500] if len(markdown_text) > 500 else markdown_text
        
        # Extraire le texte autour de l'image
        start_idx = max(0, img_match.start() - 250)
        end_idx = min(len(markdown_text), img_match.end() + 250)
        
        return markdown_text[start_idx:end_idx]


def process_pdf_with_mistral_ocr(pdf_path: str, api_key: str, image_output_dir: str = "./images_extraites", image_min_size: int = 100) -> Dict[str, Any]:
    """
    Fonction utilitaire pour traiter un PDF avec Mistral OCR.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        api_key: Clé API pour Mistral AI
        image_output_dir: Répertoire pour sauvegarder les images extraites
        image_min_size: Taille minimale des images à traiter (en pixels)
        
    Returns:
        Dictionnaire contenant le texte, les métadonnées et les images extraites
    """
    processor = MistralOCRProcessor(
        api_key=api_key,
        image_output_dir=image_output_dir,
        image_min_size=image_min_size
    )
    
    return processor.process_pdf(pdf_path) 