try:
    import os
    import re
    import io
    import base64
    import hashlib
    from typing import List, Dict, Any, Optional
    import fitz  # PyMuPDF
    import chromadb
    from chromadb.utils import embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from PIL import Image
    from mistralai import Mistral
except ImportError:
    import pip
    pip.main(['install', 'pymupdf', 'chromadb', 'langchain', 'langchain-community', 'sentence-transformers', 'pillow', 'mistralai', 'datauri'])
    
    import os
    import re
    import io
    import base64
    import hashlib
    from typing import List, Dict, Any, Optional
    import fitz  # PyMuPDF
    import chromadb
    from chromadb.utils import embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from PIL import Image
    from mistralai import Mistral

# Import conditionnel de notre module Mistral OCR
try:
    from utils.mistral_ocr import process_pdf_with_mistral_ocr
    MISTRAL_OCR_AVAILABLE = True
except ImportError:
    MISTRAL_OCR_AVAILABLE = False

class PDFProcessor:
    def __init__(
        self, 
        chroma_path: str = "./chroma_db",
        collection_name: str = "pdf_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        extract_images: bool = True,
        image_min_size: int = 100,  # Minimum size in pixels (width or height)
        store_image_content: bool = False,  # Whether to store base64 image content
        use_mistral_ocr: bool = False,
        mistral_api_key: Optional[str] = None,
        image_output_dir: str = "./images_extraites"
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chroma_path: Path to store the ChromaDB
            collection_name: Name of the collection in ChromaDB
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            model_name: HuggingFace model for embeddings
            extract_images: Whether to extract images from PDFs
            image_min_size: Minimum size for extracted images
            store_image_content: Store base64 encoded image content in metadata
            use_mistral_ocr: Whether to use Mistral OCR for processing PDFs
            mistral_api_key: API key for Mistral AI (required if use_mistral_ocr=True)
            image_output_dir: Directory to save extracted images when using Mistral OCR
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.image_min_size = image_min_size
        self.store_image_content = store_image_content
        self.use_mistral_ocr = use_mistral_ocr
        self.mistral_api_key = mistral_api_key
        self.image_output_dir = image_output_dir
        
        # Verify Mistral OCR settings if enabled
        if self.use_mistral_ocr:
            if not MISTRAL_OCR_AVAILABLE:
                raise ImportError("Mistral OCR module not available. Make sure you have installed the required packages.")
            if not self.mistral_api_key:
                raise ValueError("Mistral API key is required when using Mistral OCR.")
        
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
    
    def _extract_images_from_pdf(self, doc) -> List[Dict[str, Any]]:
        """
        Extract images from PDF document with improved detection.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of dictionaries with image info
        """
        images = []
        
        # First, try to extract using the standard method (get_images)
        for page_num, page in enumerate(doc):
            # Get list of image objects on the page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # Image reference number
                
                try:
                    # Extract image
                    base_img = doc.extract_image(xref)
                    image_bytes = base_img["image"]
                    
                    # Get image format and create PIL image
                    image_format = base_img["ext"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip small images
                    if pil_image.width < self.image_min_size or pil_image.height < self.image_min_size:
                        continue
                    
                    # Check if image has an associated mask
                    smask = img_info[1]  # Position 1 contains the smask xref
                    has_mask = smask > 0
                    
                    # Check if image is likely to be a graph/chart based on size and aspect ratio
                    is_graph = self._is_likely_graph(pil_image)
                    
                    # Get image location on page
                    image_rects = []
                    for rect in page.get_image_rects(xref):
                        image_rects.append({
                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1
                        })
                    
                    # Get surrounding text for context
                    context_text = ""
                    image_caption = ""
                    if image_rects:
                        rect = fitz.Rect(image_rects[0]["x0"], image_rects[0]["y0"], 
                                          image_rects[0]["x1"], image_rects[0]["y1"])
                        
                        # Get surrounding text
                        rect = fitz.Rect(rect.x0 - 50, rect.y0 - 50, rect.x1 + 50, rect.y1 + 50)
                        context_text = page.get_text("text", clip=rect)
                        
                        # Try to find caption below the image
                        caption_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 100)
                        image_caption = page.get_text("text", clip=caption_rect)
                        
                        # If caption is long, try to limit it to the first paragraph
                        if len(image_caption) > 100:
                            caption_parts = image_caption.split('\n\n')
                            if len(caption_parts) > 1:
                                image_caption = caption_parts[0]
                    
                    # Use hash of image bytes as unique identifier
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    # Determine image type (figure, graph, chart, etc.)
                    img_type = "image"
                    
                    # Check for keywords in caption or surrounding text that suggest this is a graph/chart/figure
                    graph_keywords = ["figure", "fig.", "chart", "graph", "plot", "diagram", "graphique", "tableau"]
                    for keyword in graph_keywords:
                        if (keyword.lower() in image_caption.lower() or 
                            keyword.lower() in context_text.lower()):
                            if "chart" in image_caption.lower() or "graph" in image_caption.lower() or is_graph:
                                img_type = "graph"
                            else:
                                img_type = "figure"
                            break
                    
                    # Create image info
                    image_info = {
                        "id": f"img_{img_hash}",
                        "page_num": page_num + 1,
                        "img_index": img_index,
                        "format": image_format,
                        "width": pil_image.width,
                        "height": pil_image.height,
                        "location": image_rects,
                        "context_text": context_text,
                        "caption": image_caption,
                        "type": img_type,
                        "description": f"{img_type.capitalize()} on page {page_num + 1}",
                        "has_mask": has_mask,
                    }
                    
                    # Add base64 encoding of image if requested
                    if self.store_image_content:
                        image_info["content"] = base64.b64encode(image_bytes).decode('utf-8')
                    
                    images.append(image_info)
                except Exception as e:
                    print(f"Error extracting image: {str(e)}")
                    continue
        
        # Alternative extraction method for figures not detected by the standard method
        # Use the get_text("dict") method which can find images embedded in different ways
        if len(images) == 0:
            try:
                for page_num, page in enumerate(doc):
                    dict_data = page.get_text("dict")
                    for block in dict_data["blocks"]:
                        if block["type"] == 1:  # Image block
                            try:
                                # Image data from the dict
                                img_bytes = block.get("image", b"")
                                if not img_bytes:
                                    continue
                                    
                                # Create PIL image
                                pil_image = Image.open(io.BytesIO(img_bytes))
                                
                                # Skip small images
                                if pil_image.width < self.image_min_size or pil_image.height < self.image_min_size:
                                    continue
                                
                                # Check if image is likely to be a graph/chart
                                is_graph = self._is_likely_graph(pil_image)
                                
                                # Extract image rectangle
                                bbox = block.get("bbox", (0, 0, 0, 0))
                                image_rect = {
                                    "x0": bbox[0],
                                    "y0": bbox[1],
                                    "x1": bbox[2],
                                    "y1": bbox[3]
                                }
                                
                                # Get format or default to JPEG
                                image_format = block.get("ext", "jpeg")
                                
                                # Get surrounding text
                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                # Créer un rectangle étendu manuellement au lieu d'utiliser expand
                                expanded_rect = fitz.Rect(
                                    rect.x0 - 50, 
                                    rect.y0 - 50,
                                    rect.x1 + 50,
                                    rect.y1 + 50
                                )
                                context_text = page.get_text("text", clip=expanded_rect)
                                
                                # Try to find caption below the image
                                caption_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 100)
                                image_caption = page.get_text("text", clip=caption_rect)
                                
                                # Generate hash
                                img_hash = hashlib.md5(img_bytes).hexdigest()
                                
                                # Determine image type
                                img_type = "image"
                                graph_keywords = ["figure", "fig.", "chart", "graph", "plot", "diagram", "graphique", "tableau"]
                                for keyword in graph_keywords:
                                    if (keyword.lower() in image_caption.lower() or 
                                        keyword.lower() in context_text.lower()):
                                        if "chart" in image_caption.lower() or "graph" in image_caption.lower() or is_graph:
                                            img_type = "graph"
                                        else:
                                            img_type = "figure"
                                        break
                                
                                # Create image info
                                image_info = {
                                    "id": f"img_{img_hash}",
                                    "page_num": page_num + 1,
                                    "img_index": len(images),
                                    "format": image_format,
                                    "width": pil_image.width,
                                    "height": pil_image.height,
                                    "location": [image_rect],
                                    "context_text": context_text,
                                    "caption": image_caption,
                                    "type": img_type,
                                    "description": f"{img_type.capitalize()} on page {page_num + 1}",
                                    "has_mask": False,
                                }
                                
                                # Add base64 encoding of image if requested
                                if self.store_image_content:
                                    image_info["content"] = base64.b64encode(img_bytes).decode('utf-8')
                                
                                images.append(image_info)
                            except Exception as e:
                                print(f"Error extracting image from dict: {str(e)}")
                                continue
            except Exception as e:
                print(f"Error in alternative image extraction: {str(e)}")
        
        return images
    
    def _is_likely_graph(self, pil_image):
        """
        Analyze image to determine if it's likely to be a graph or chart.
        Basic heuristic based on color count and edges.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Boolean indicating if image is likely a graph
        """
        try:
            # Convert to grayscale for analysis
            gray_img = pil_image.convert('L')
            
            # Count distinct colors (limited sample for performance)
            # Most graphs have a limited color palette
            colors = set()
            width, height = pil_image.size
            
            # Sample pixels at regular intervals
            for x in range(0, width, max(1, width // 100)):
                for y in range(0, height, max(1, height // 100)):
                    colors.add(pil_image.getpixel((x, y)))
            
            # Graphs typically have a limited color palette
            if len(colors) < 50:
                return True
                
            # Simple edge detection - graphs typically have many edges
            # Use basic filtering
            from PIL import ImageFilter
            
            # Apply edge detection filter
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            
            # Count pixels that are likely edges
            edge_pixels = 0
            total_pixels = width * height
            
            for x in range(0, width, max(1, width // 50)):
                for y in range(0, height, max(1, height // 50)):
                    if edges.getpixel((x, y)) > 128:  # Bright pixels are edges
                        edge_pixels += 1
            
            # Calculate edge ratio
            edge_ratio = edge_pixels / (width * height / 2500)
            
            # Graphs typically have a higher edge ratio
            if edge_ratio > 0.1:
                return True
                
            return False
        except Exception:
            # If analysis fails, don't classify as graph
            return False
    
    def extract_text_and_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text and metadata
        """
        if self.use_mistral_ocr and self.mistral_api_key:
            # Use Mistral OCR to process the PDF
            try:
                ocr_result = process_pdf_with_mistral_ocr(pdf_path, self.mistral_api_key)
                return ocr_result
            except Exception as e:
                print(f"Error with Mistral OCR, falling back to PyMuPDF: {str(e)}")
        
        # If not using Mistral OCR or if it failed, use PyMuPDF
        return self._extract_with_pymupdf(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata using PyMuPDF.
        
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
        
        # Extract images if requested
        images = []
        if self.extract_images:
            images = self._extract_images_from_pdf(doc)
            metadata["has_images"] = len(images) > 0
            metadata["image_count"] = len(images)
        
        doc.close()
        return {"text": text, "metadata": metadata, "images": images}
    
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
        images = data["images"]
        
        # Create chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add chunks to ChromaDB
        doc_ids = []
        
        # Process text chunks
        for i, chunk in enumerate(chunks):
            # Create unique ID for the chunk
            doc_id = f"{os.path.basename(pdf_path)}_txt_{i}"
            doc_ids.append(doc_id)
            
            # Create metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["content_type"] = "text"
            
            # Get embedding for the chunk
            embedding = self.embeddings.embed_query(chunk)
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
                documents=[chunk]
            )
        
        # Process images
        for i, img_info in enumerate(images):
            try:
                # Extract image context and create a description
                context = img_info.get("context_text", "")
                img_type = img_info.get("type", "image")
                caption = img_info.get("caption", "")
                
                # Create description based on available information
                if caption:
                    description = f"{img_type.capitalize()} on page {img_info['page_num']}: {caption}"
                else:
                    description = f"{img_type.capitalize()} on page {img_info['page_num']}: {context[:200]}"
                
                # Create unique ID for the image
                doc_id = f"{os.path.basename(pdf_path)}_img_{i}"
                doc_ids.append(doc_id)
                
                # Create metadata for this image
                img_metadata = metadata.copy()
                img_metadata.update({
                    "content_type": "image",
                    "type": img_type,
                    "page_num": img_info["page_num"],
                    "img_width": img_info.get("width", 0),
                    "img_height": img_info.get("height", 0),
                    "img_format": img_info.get("format", "jpeg"),
                    "caption": caption
                })
                
                # If using Mistral OCR, add path to saved image
                if self.use_mistral_ocr and "path" in img_info:
                    img_metadata["img_path"] = img_info["path"]
                
                # If storing image content and it's available, add it to metadata
                if self.store_image_content and "content" in img_info:
                    img_metadata["img_content"] = img_info["content"]
                
                # Get embedding for the image context
                embedding = self.embeddings.embed_query(description)
                
                # Add to collection
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[img_metadata],
                    documents=[description]
                )
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
        
        return doc_ids
    
    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files
            
        Returns:
            Dictionary mapping filenames to lists of document IDs
        """
        # Validation
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all PDF files in the directory
        pdf_files = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, filename))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return {}
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF file
        results = {}
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"Processing {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
                
                # If using Mistral OCR, create a dedicated output directory for this file
                if self.use_mistral_ocr:
                    file_image_dir = os.path.join(self.image_output_dir, os.path.splitext(os.path.basename(pdf_path))[0])
                    if not os.path.exists(file_image_dir):
                        os.makedirs(file_image_dir, exist_ok=True)
                
                # Process the PDF
                doc_ids = self.process_pdf(pdf_path)
                
                # Store the results
                results[os.path.basename(pdf_path)] = doc_ids
                
                print(f"  Added {len(doc_ids)} chunks to database")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                results[os.path.basename(pdf_path)] = []
                continue
        
        return results
    
    def search(self, query: str, n_results: int = 5, filter_by: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Search the database for documents matching the query.
        
        Args:
            query: Query string
            n_results: Number of results to return
            filter_by: Dictionary of metadata filters
            
        Returns:
            List of dictionaries containing text and metadata
        """
        # Get embedding for the query
        embedding = self.embeddings.embed_query(query)
        
        # Prepare where document for filtering
        where_document = {}
        if filter_by:
            for key, value in filter_by.items():
                where_document[key] = value
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_document if where_document else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
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
    
    # Filter search example
    # text_only_results = processor.search("your search query", filter_by={"content_type": "text"})
    # image_only_results = processor.search("your search query", filter_by={"content_type": "image"})
