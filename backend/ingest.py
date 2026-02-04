from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pdfplumber

from .rag import Chunk, save_index

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
HTML_DIR = STORAGE_DIR / "html"

BRANDS = {"aspirapolvere", "condizionatore", "congelatore", "lavastoviglie", "lavatrice", "microonde", "plasma"}

LOG_DIR = STORAGE_DIR / "logs"
LOG_FILE = LOG_DIR / "ingest.log"
CAPTIONS_FILE = STORAGE_DIR / "image_captions.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


@dataclass
class IngestResult:
    manuals: int
    chunks: int
    images: int


def infer_brand(file_name: str) -> str:
    lower = file_name.lower()
    for brand in BRANDS:
        if brand in lower:
            return brand
    return "unknown"


def list_manual_pdfs() -> List[Path]:
    candidates: List[Path] = []
    manuals_dir = BASE_DIR / "manuals"
    if manuals_dir.exists():
        candidates.extend(manuals_dir.glob("*.pdf"))
    candidates.extend(BASE_DIR.glob("*.pdf"))
    unique = {p.resolve() for p in candidates if p.is_file()}
    return sorted(unique)


def chunk_text(text: str, size: int = 1000, overlap: int = 0) -> List[str]:
    logger.debug("chunk_text: inizio normalizzazione")
    if not text or not text.strip():
        return []

    title_pattern = re.compile(r"^\d+(?:\.\d+)*\.?\s+.+")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    paragraphs: List[str] = []
    current: List[str] = []
    pending_title = ""

    for line in lines:
        if title_pattern.match(line):
            if current:
                paragraph_text = " ".join(current).strip()
                if pending_title:
                    paragraph_text = f"{pending_title}\n{paragraph_text}"
                    pending_title = ""
                paragraphs.append(paragraph_text)
                current = []
            pending_title = line
            continue

        # Se il titolo è spezzato su più righe, unisci la riga successiva
        if pending_title and not current and not title_pattern.match(line):
            if len(line) <= 60 or (line and line[0].islower()):
                pending_title = f"{pending_title} {line}".strip()
                continue

        current.append(line)

    if current:
        paragraph_text = " ".join(current).strip()
        if pending_title:
            paragraph_text = f"{pending_title}\n{paragraph_text}"
            pending_title = ""
        paragraphs.append(paragraph_text)
    elif pending_title:
        paragraphs.append(pending_title)

    # Fallback: se non abbiamo paragrafi, normalizza tutto
    if not paragraphs:
        normalized = re.sub(r"\s+", " ", text).strip()
        return [normalized] if normalized else []

    chunks: List[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if not paragraph:
            continue

        if len(paragraph) > size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            # Mantieni il titolo con il testo successivo quando possibile
            if "\n" in paragraph:
                title_line, body = paragraph.split("\n", 1)
                title_line = title_line.strip()
                body = body.strip()
                if len(title_line) + 1 < size:
                    start = 0
                    first_chunk = f"{title_line}\n"
                    remaining_space = size - len(first_chunk)
                    first_piece = body[:remaining_space].strip()
                    if first_piece:
                        chunks.append(first_chunk + first_piece)
                    start = remaining_space
                    while start < len(body):
                        end = min(start + size, len(body))
                        piece = body[start:end].strip()
                        if piece:
                            chunks.append(piece)
                        start = end
                else:
                    chunks.append(paragraph[:size].strip())
            else:
                start = 0
                while start < len(paragraph):
                    end = min(start + size, len(paragraph))
                    piece = paragraph[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    start = end
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = paragraph

    if buffer:
        chunks.append(buffer.strip())

    logger.debug("chunk_text: fine - %d chunk creati", len(chunks))
    return chunks


def extract_vector_graphics_as_images(page, page_num: int, pdf_path: Path, brand: str) -> List[str]:
    """Estrae elementi vettoriali dalla pagina e li rende come immagini separate."""
    import fitz
    
    drawings = page.get_drawings()
    if not drawings:
        return []
    
    # Raggruppa drawings per prossimità (clustering semplice)
    clusters = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if not rect:
            continue
        
        # Cerca cluster esistente vicino
        merged = False
        for cluster in clusters:
            cluster_rect = cluster["rect"]
            # Espandi leggermente per catturare elementi vicini
            expanded = fitz.Rect(
                cluster_rect.x0 - 5, cluster_rect.y0 - 5,
                cluster_rect.x1 + 5, cluster_rect.y1 + 5
            )
            if expanded.intersects(rect) or expanded.contains(rect):
                cluster["rect"] = cluster_rect | rect  # Union dei rect
                cluster["count"] += 1
                merged = True
                break
        
        if not merged:
            clusters.append({"rect": rect, "count": 1})
    
    # Filtra cluster troppo piccoli (probabilmente decorazioni)
    MIN_SIZE = 50  # pixel
    MIN_AREA = 2500  # pixel quadrati
    
    significant_clusters = [
        c for c in clusters 
        if (c["rect"].width >= MIN_SIZE and c["rect"].height >= MIN_SIZE) 
        or (c["rect"].width * c["rect"].height >= MIN_AREA)
    ]
    
    logger.debug("Pagina %d: %d drawings → %d clusters → %d significativi", 
                page_num, len(drawings), len(clusters), len(significant_clusters))
    
    # Rendi ogni cluster come immagine
    extracted_images = []
    for idx, cluster in enumerate(significant_clusters, start=1):
        rect = cluster["rect"]
        
        # Aggiungi margine
        margin = 10
        clip_rect = fitz.Rect(
            max(0, rect.x0 - margin),
            max(0, rect.y0 - margin),
            min(page.rect.width, rect.x1 + margin),
            min(page.rect.height, rect.y1 + margin)
        )
        
        # Valida clip_rect
        if clip_rect.is_empty or clip_rect.width < 10 or clip_rect.height < 10:
            logger.debug("  Cluster %d troppo piccolo: %dx%d px, skipped", 
                        idx, int(clip_rect.width), int(clip_rect.height))
            continue
        
        try:
            # Rendi la porzione di pagina (senza matrix per evitare overflow)
            pix = page.get_pixmap(clip=clip_rect, alpha=False)
            
            # Salva immagine
            img_name = f"{brand}_{pdf_path.stem}_p{page_num}_vec{idx}.png"
            img_path = IMAGES_DIR / img_name
            pix.save(str(img_path))
            
            img_url = f"/static/images/{img_name}"
            extracted_images.append(img_url)
            
            logger.debug("  Estratto vettoriale %d: %dx%d px → %s", 
                        idx, int(rect.width), int(rect.height), img_name)
        except Exception as e:
            logger.warning("  Errore rendering vettoriale %d: %s", idx, str(e))
            continue
    
    return extracted_images


def pdf_to_html_with_images(pdf_path: Path) -> Tuple[str, Path, Dict[int, List[str]]]:
    """Converte PDF a HTML ed estrae immagini (raster + vettoriali). Ritorna (html_content, html_file, images_by_page)"""
    import fitz
    
    html_dir = Path("backend/storage/html")
    html_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    html_file = html_dir / f"{pdf_path.stem}_raw.html"
    
    try:
        
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html>")
        html_parts.append("<head>")
        html_parts.append(f"<title>{pdf_path.stem}</title>")
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append("</head>")
        html_parts.append("<body>")
        
        doc = fitz.open(pdf_path)
        logger.info("Apertura PDF: %s - %d pagine", pdf_path.name, len(doc))
        
        images_by_page = {}
        brand = infer_brand(pdf_path.name)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_index = page_num + 1
            
            html_parts.append(f'<a name="page{page_index}"></a>')
            html_parts.append(f"<h2>Pagina {page_index}</h2>")
            
            # Estrai testo HTML
            page_html = page.get_text("html")
            html_parts.append(page_html)
            
            # 1. Estrai immagini RASTER
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list, start=1):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    img_name = f"{brand}_{pdf_path.stem}_p{page_index}_img{img_index}.{image_ext}"
                    img_path = IMAGES_DIR / img_name
                    
                    img_path.write_bytes(image_bytes)
                    
                    img_url = f"/static/images/{img_name}"
                    images_by_page.setdefault(page_index, []).append(img_url)
                    
                    html_parts.append(f'<img src="{img_url}" alt="Immagine raster {img_index}" />')
                    
                    logger.debug("Immagine raster: %s (pag %d)", img_name, page_index)
                except Exception as e:
                    logger.warning("Errore estrazione raster xref=%d pag %d: %s", xref, page_index, str(e))
            
            # 2. Estrai elementi VETTORIALI
            vector_images = extract_vector_graphics_as_images(page, page_index, pdf_path, brand)
            for vec_url in vector_images:
                images_by_page.setdefault(page_index, []).append(vec_url)
                html_parts.append(f'<img src="{vec_url}" alt="Grafico vettoriale" />')
            
            if page_index % 5 == 0:
                total_imgs = sum(len(v) for v in images_by_page.values())
                logger.info("Processata pagina %d/%d (%d immagini totali)", page_index, len(doc), total_imgs)
        
        doc.close()
        
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        html_content = "\n".join(html_parts)
        
        # Salva file HTML
        html_file.write_text(html_content, encoding="utf-8")
        total_images = sum(len(v) for v in images_by_page.values())
        logger.info("HTML salvato: %s (%d immagini estratte: raster + vettoriali)", html_file, total_images)
        
        return html_content, html_file, images_by_page
    
    except Exception as e:
        logger.error("Errore nella conversione PDF a HTML: %s", str(e))
        raise


def pdf_to_html(pdf_path: Path) -> Tuple[str, Path]:
    """Wrapper per mantenere compatibilità"""
    html_content, html_file, _ = pdf_to_html_with_images(pdf_path)
    return html_content, html_file



def parse_html_for_content(html_content: str, pdf_path: Path, html_dir: Path) -> Tuple[List[Tuple[str, int]], Dict[int, List[str]]]:
    """Parsea HTML e estrae testo + immagini. Ritorna (lista_paragrafi_con_pagina, immagini_per_pagina)"""
    from bs4 import BeautifulSoup
    import base64
    import re
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    paragraphs_with_page = []
    images_by_page = {}
    
    # Estrai pagine (cercando <a name="page*"></a>)
    current_page = 1
    current_text = []
    image_counter = {}
    
    for element in soup.find_all(["a", "p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "img", "pre", "span"]):
        # Rileva cambio pagina
        if element.name == "a" and element.get("name", "").startswith("page"):
            if current_text:
                text = "\n".join(current_text).strip()
                if text:
                    paragraphs_with_page.append((text, current_page))
                current_text = []
            
            try:
                current_page = int(element.get("name", "page1").replace("page", ""))
            except:
                current_page += 1
            continue
        
        # Estrai immagini (PyMuPDF le embedded come data URI base64)
        if element.name == "img":
            img_src = element.get("src", "")
            if img_src.startswith("data:image"):
                # Decodifica data URI e salva come file
                try:
                    # Formato: data:image/png;base64,iVBORw0KG...
                    match = re.match(r"data:image/(\w+);base64,(.+)", img_src)
                    if match:
                        img_format = match.group(1)
                        img_data = base64.b64decode(match.group(2))
                        
                        brand = infer_brand(pdf_path.name)
                        img_num = image_counter.get(current_page, 0) + 1
                        image_counter[current_page] = img_num
                        
                        img_name = f"{brand}_{pdf_path.stem}_p{current_page}_{img_num}.{img_format}"
                        img_dest = IMAGES_DIR / img_name
                        
                        # Salva il file
                        img_dest.write_bytes(img_data)
                        
                        # Registra nell'HTML e in images_by_page
                        img_url = f"/static/images/{img_name}"
                        images_by_page.setdefault(current_page, []).append(img_url)
                        
                        # Aggiungi riferimento nel testo
                        current_text.append(f"[IMMAGINE: {img_url}]")
                        logger.debug("Immagine estratta: %s (pag %d)", img_name, current_page)
                except Exception as e:
                    logger.warning("Errore nell'estrarre immagine dalla pagina %d: %s", current_page, str(e))
            continue
        
        # Estrai testo
        text = element.get_text(strip=True)
        if text and len(text) > 3:  # Ignora testo troppo corto
            current_text.append(text)
    
    # Aggiungi ultimo paragrafo
    if current_text:
        text = "\n".join(current_text).strip()
        if text:
            paragraphs_with_page.append((text, current_page))
    
    logger.info("HTML parsato: %d paragrafi, %d immagini da %d pagine", 
                len(paragraphs_with_page), sum(len(v) for v in images_by_page.values()), len(images_by_page))
    
    return paragraphs_with_page, images_by_page



def generate_image_captions() -> Dict[str, str]:
    """Crea placeholder per le didascalie. L'utente edita manualmente con caption_review.bat"""
    
    # Carica captions esistenti
    captions = load_image_captions()
    logger.info("Captions esistenti caricate: %d", len(captions))
    
    image_files = list(IMAGES_DIR.glob("*.png"))
    logger.info("Trovate %d immagini totali", len(image_files))
    
    if not image_files:
        logger.warning("Nessuna immagine trovata in %s", IMAGES_DIR)
        return captions
    
    # Filtra solo le immagini che non hanno già una caption
    images_to_process = []
    for image_path in image_files:
        image_url = f"/static/images/{image_path.name}"
        if image_url not in captions:
            images_to_process.append(image_path)
    
    logger.info("Immagini nuove (senza caption): %d", len(images_to_process))
    
    if not images_to_process:
        logger.info("Tutte le immagini hanno già una caption")
        return captions
    
    # Assegna placeholder a tutte le immagini nuove
    for image_path in images_to_process:
        image_url = f"/static/images/{image_path.name}"
        captions[image_url] = "Immagine del manuale"
    
    # Salva captions in JSON
    CAPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS_FILE.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Captions placeholder salvate in %s (%d immagini)", CAPTIONS_FILE, len(captions))
    print(f"\n✓ Placeholder captions salvati: {CAPTIONS_FILE}")
    print("Per personalizzare le didascalie, esegui: python -m backend.caption_review")
    
    return captions


def load_image_captions() -> Dict[str, str]:
    """Carica le didascalie salvate delle immagini."""
    if CAPTIONS_FILE.exists():
        try:
            return json.loads(CAPTIONS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Errore caricamento captions: %s", str(e))
    return {}


def save_captions(captions: Dict[str, str]) -> None:
    CAPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS_FILE.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_images(pdf_path: Path) -> Dict[int, List[str]]:
    """Estrae immagini dal PDF usando pdf2image (cattura anche vettoriali)."""
    from pdf2image import convert_from_path
    from PIL import Image
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    page_images: Dict[int, List[str]] = {}
    
    logger.info("=== Analisi PDF: %s ===", pdf_path.name)
    
    try:
        # Converti ogni pagina in immagine
        images = convert_from_path(pdf_path, dpi=150)  # 150 DPI è buon compromesso qualità/dimensione
        logger.info("Totale pagine: %d", len(images))
        
        brand = infer_brand(pdf_path.name)
        
        for page_index, image in enumerate(images, start=1):
            # Salva ogni pagina come immagine
            file_name = f"{brand}_{pdf_path.stem}_p{page_index}.png"
            image_path = IMAGES_DIR / file_name
            
            if image_path.exists():
                logger.debug("Immagine già estratta: %s", file_name)
                relative = f"/static/images/{file_name}"
                page_images.setdefault(page_index, []).append(relative)
                continue
            
            try:
                image.save(image_path, "PNG")
                width, height = image.size
                logger.info("Pagina %d: estratta immagine %dx%d", page_index, width, height)
                
                relative = f"/static/images/{file_name}"
                page_images.setdefault(page_index, []).append(relative)
            except Exception as e:
                logger.error("Errore nel salvare pagina %d: %s", page_index, str(e))
                continue
    
    except Exception as e:
        logger.error("Errore nell'estrazione delle immagini da %s: %s", pdf_path.name, str(e))
        return page_images
    
    total_extracted = sum(len(v) for v in page_images.values())
    logger.info("PDF %s: estratte %d immagini totali (1 per pagina)", pdf_path.name, total_extracted)
    return page_images


def cleanup_small_images(min_width: int = 80, min_height: int = 80, min_area: int = 8000) -> int:
    import fitz
    
    if not IMAGES_DIR.exists():
        return 0

    captions = load_image_captions()
    removed = 0

    for image_path in IMAGES_DIR.glob("*.png"):
        try:
            pix = fitz.Pixmap(str(image_path))
            width = pix.width
            height = pix.height
            pix = None
        except Exception:
            width = 0
            height = 0

        if width < min_width or height < min_height or (width * height) < min_area:
            try:
                image_path.unlink()
                removed += 1
            except Exception:
                pass

            image_url = f"/static/images/{image_path.name}"
            if image_url in captions:
                captions.pop(image_url, None)

    if removed:
        save_captions(captions)

    return removed


def build_html(pdf_path: Path, chunks_by_page: Dict[int, List[Chunk]], images_by_page: Dict[int, List[str]]) -> str:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    html_path = HTML_DIR / f"{pdf_path.stem}.html"
    
    # Carica le captions delle immagini
    captions = load_image_captions()

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>',
        f"<title>{html.escape(pdf_path.stem)}</title>",
        """<style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f5;
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                overflow: hidden;
            }
            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }
            h1 {
                font-size: 2em;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .page {
                padding: 30px;
                border-bottom: 1px solid #e0e0e0;
                position: relative;
            }
            .page:last-child {
                border-bottom: none;
            }
            .page-indicator {
                position: absolute;
                top: 10px;
                right: 15px;
                font-size: 0.75em;
                color: #999;
                font-weight: normal;
            }
            .page-header {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 3px solid #667eea;
            }
            .page-number {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                margin-right: 15px;
                font-size: 0.9em;
            }
            h2 {
                color: #333;
                font-size: 1.5em;
                font-weight: 600;
            }
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Consolas', 'Monaco', monospace;
                line-height: 1.7;
                color: #444;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                font-size: 0.95em;
            }
            .heading-1 {
                font-weight: bold;
                font-size: 1.3em;
                color: #667eea;
                margin: 20px 0 10px 0;
                display: block;
            }
            .heading-2 {
                font-weight: bold;
                font-size: 1.15em;
                color: #764ba2;
                margin: 15px 0 8px 0;
                display: block;
            }
            .chunk {
                background: #fafafa;
                border-left: 3px solid #667eea;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .chunk-header {
                font-size: 0.7em;
                color: #999;
                margin-bottom: 10px;
                font-weight: 600;
            }
            .chunk-text p {
                margin: 0 0 8px 0;
                line-height: 1.7;
                color: #444;
            }
            .chunk-text ul {
                margin: 8px 0 12px 20px;
                padding: 0;
            }
            .chunk-text li {
                margin: 4px 0;
            }
            .toc {
                background: #f8f9fa;
                padding: 30px;
                border-bottom: 2px solid #e0e0e0;
            }
            .toc h3 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            .toc ul {
                list-style: none;
            }
            .toc li {
                margin: 8px 0;
            }
            .toc a {
                color: #667eea;
                text-decoration: none;
                padding: 5px 10px;
                display: inline-block;
                border-radius: 5px;
                transition: all 0.3s;
            }
            .toc a:hover {
                background: #667eea;
                color: white;
                transform: translateX(5px);
            }
            .images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 25px;
            }
            .image-item {
                background: white;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #e0e0e0;
                transition: transform 0.3s;
            }
            .image-item:hover {
                transform: translateY(-5px);
            }
            .image-item img {
                width: 100%;
                height: auto;
                display: block;
            }
            .image-caption {
                padding: 12px;
                background: #f8f9fa;
                font-size: 0.85em;
                color: #555;
                line-height: 1.5;
                border-top: 1px solid #e0e0e0;
            }
        </style>""",
        "</head>",
        "<body>",
        '<div class="container">',
        "<header>",
        f"<h1>📄 {html.escape(pdf_path.stem)}</h1>",
        f"<p style='opacity:0.9; margin-top:10px;'>{len(chunks_by_page)} pagine</p>",
        "</header>",
        '<div class="toc">',
        '<h3>📑 Indice</h3>',
        '<ul>',
    ]

    # Genera indice
    for idx in sorted(chunks_by_page.keys()):
        parts.append(f'<li><a href="#page-{idx}">Pagina {idx}</a></li>')
    
    parts.append('</ul>')
    parts.append('</div>')

    # Genera pagine con chunk
    for page_num in sorted(chunks_by_page.keys()):
        page_chunks = chunks_by_page[page_num]
        parts.append(f'<div class="page" id="page-{page_num}">')
        parts.append(f'<div class="page-indicator">Pag. {page_num}</div>')
        parts.append('<div class="page-header">')
        parts.append(f'<span class="page-number">Pag. {page_num}</span>')
        parts.append(f'<h2>Pagina {page_num}</h2>')
        parts.append('</div>')
        
        # Mostra ogni chunk
        for idx, chunk in enumerate(page_chunks, 1):
            parts.append(f'<div class="chunk" id="chunk-{chunk.id}">')
            parts.append(f'<div class="chunk-header">CHUNK {idx}/{len(page_chunks)}</div>')
            
            # Rimuovi le note sulle immagini dal testo per l'HTML
            chunk_text = chunk.text
            if "[Immagini disponibili in questa pagina:" in chunk_text:
                chunk_text = chunk_text.split("[Immagini disponibili in questa pagina:")[0].strip()
            
            # Formatta il testo e gli elenchi puntati
            lines = [ln.strip() for ln in chunk_text.split('\n') if ln.strip()]
            html_parts = []
            list_items = []

            def flush_list() -> None:
                nonlocal list_items
                if list_items:
                    html_parts.append("<ul>")
                    html_parts.extend(list_items)
                    html_parts.append("</ul>")
                    list_items = []

            for line in lines:
                # Titoli numerati
                if re.match(r'^\d+\.(\d+\.)*\s', line):
                    flush_list()
                    level = line.split()[0].count('.')
                    css_class = "heading-1" if level <= 1 else "heading-2"
                    safe_line = html.escape(line)
                    html_parts.append(f'<div class="{css_class}">{safe_line}</div>')
                    continue

                # Elenchi puntati
                if re.match(r'^[-•*]\s+', line):
                    item_text = re.sub(r'^[-•*]\s+', '', line)
                    list_items.append(f"<li>{html.escape(item_text)}</li>")
                    continue

                flush_list()
                html_parts.append(f"<p>{html.escape(line)}</p>")

            flush_list()

            formatted_text = "\n".join(html_parts) if html_parts else f"<p>{html.escape(chunk_text)}</p>"
            parts.append(f"<div class=\"chunk-text\">{formatted_text}</div>")
            parts.append('</div>')  # chiudi chunk
        
        # Aggiungi immagini della pagina
        if page_num in images_by_page and images_by_page[page_num]:
            parts.append('<div class="images-grid">')
            for img_url in images_by_page[page_num]:
                img_relative = f"../images/{Path(img_url).name}"
                caption = captions.get(img_url, "Immagine del manuale")
                parts.append('<div class="image-item">')
                parts.append(f'<img src="{img_relative}" alt="{html.escape(caption)}" loading="lazy"/>')
                parts.append(f'<div class="image-caption">{html.escape(caption)}</div>')
                parts.append('</div>')
            parts.append('</div>')
        
        parts.append("</div>")  # chiudi page

    parts.append("</div>")
    parts.append("</body>")
    parts.append("</html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")
    return f"/static/html/{html_path.name}"


def extract_text_with_structure(pdf_path: Path) -> List[str]:
    """Estrae testo preservando struttura (titoli, paragrafi) usando pymupdf."""
    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        blocks = page.get_text("blocks")
        
        page_content: List[str] = []
        for block in blocks:
            if block[6] == 0:  # tipo 0 = blocco di testo
                text = block[4].strip()
                if text:
                    page_content.append(text)
        
        pages_text.append("\n".join(page_content))
    
    doc.close()
    return pages_text


def _ingest_pdf_legacy(pdf_path: Path, brand: str, images_by_page: Dict[int, List[str]], captions: Dict[str, str]) -> Tuple[List[Chunk], int]:
    """Fallback per estrazione diretta da PDF quando HTML parsing fallisce"""
    chunks: List[Chunk] = []
    pages_text = extract_text_with_structure(pdf_path)
    logger.info("Fallback: Estratto testo da %s pagine con struttura diretta", len(pages_text))

    for page_index, text in enumerate(pages_text, start=1):
        logger.info("Chunking pagina %d (fallback) - testo: %d char", page_index, len(text))
        
        # Estrai i paragrafi mantenendo la struttura
        title_pattern = re.compile(r"^\d+(?:\.\d+)*\.?\s+.+")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        paragraphs: List[str] = []
        current: List[str] = []
        pending_title = ""

        for line in lines:
            if title_pattern.match(line):
                if current:
                    paragraph_text = " ".join(current).strip()
                    if pending_title:
                        paragraph_text = f"{pending_title}\n{paragraph_text}"
                        pending_title = ""
                    paragraphs.append(paragraph_text)
                    current = []
                pending_title = line
                continue

            if pending_title and not current and not title_pattern.match(line):
                if len(line) <= 60 or (line and line[0].islower()):
                    pending_title = f"{pending_title} {line}".strip()
                    continue

            current.append(line)

        if current:
            paragraph_text = " ".join(current).strip()
            if pending_title:
                paragraph_text = f"{pending_title}\n{paragraph_text}"
                pending_title = ""
            paragraphs.append(paragraph_text)
        elif pending_title:
            paragraphs.append(pending_title)

        # Per ogni paragrafo, crea la versione con immagini embedded
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Crea una versione del paragrafo con le immagini incorporate
            full_paragraph_with_images = paragraph
            page_images = images_by_page.get(page_index, [])
            if page_images:
                # Aggiungi le immagini alla fine del paragrafo con le loro didascalie
                image_blocks = []
                for img_url in page_images:
                    caption = captions.get(img_url, "Immagine illustrativa")
                    image_blocks.append(f"[IMMAGINE: {img_url} - {caption}]")
                full_paragraph_with_images = f"{paragraph}\n\n" + "\n".join(image_blocks)

            # Ora chunka il paragrafo se è troppo lungo
            pieces = chunk_text(paragraph, size=1000, overlap=0)
            if not pieces:
                pieces = [paragraph]

            # Per ogni chunk del paragrafo, crea un Chunk con il full_paragraph completo
            for piece in pieces:
                chunk_id = str(uuid.uuid4())
                html_file = ""
                html_anchor = f"chunk-{chunk_id}"
                chunk = Chunk(
                    id=chunk_id,
                    brand=brand,
                    manual=pdf_path.stem,
                    page=page_index,
                    text=piece,  # Il chunk segmentato per la ricerca
                    images=page_images,
                    html_file=html_file,
                    html_anchor=html_anchor,
                    full_paragraph=full_paragraph_with_images,  # Paragrafo completo con immagini per l'LLM
                )
                chunks.append(chunk)
        
        if page_index % 5 == 0 or page_index == 1:
            logger.info("Elaborata pagina %d/%d (fallback) - %d chunk creati", page_index, len(pages_text), len(chunks))

    # Raggruppa chunk per pagina per l'HTML
    chunks_by_page: Dict[int, List[Chunk]] = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page, []).append(chunk)

    logger.info("Generazione HTML per %s (fallback)", pdf_path.name)
    html_file = build_html(pdf_path, chunks_by_page, images_by_page)
    for chunk in chunks:
        chunk.html_file = html_file

    logger.info("Creati %s chunk per %s (fallback)", len(chunks), pdf_path.name)
    return chunks, sum(len(v) for v in images_by_page.values())


def ingest_pdf(pdf_path: Path) -> Tuple[List[Chunk], int]:
    brand = infer_brand(pdf_path.name)
    logger.info("Inizio ingestione manuale %s (brand=%s)", pdf_path.name, brand)
    
    # Step 1: Converti PDF a HTML ED estrai immagini
    try:
        html_content, html_file, images_by_page = pdf_to_html_with_images(pdf_path)
    except Exception as e:
        logger.error("Fallito il parsing HTML per %s: %s. Fallback a estrazione diretta.", pdf_path.name, str(e))
        # Fallback: usa il vecchio metodo
        images_by_page = extract_images(pdf_path)
        captions = load_image_captions()
        return _ingest_pdf_legacy(pdf_path, brand, images_by_page, captions)
    
    # Step 2: Parsea HTML per estrarre testo
    try:
        html_dir = html_file.parent
        paragraphs_with_page, images_from_html = parse_html_for_content(html_content, pdf_path, html_dir)
        
        # Le immagini sono già state estratte da pdf_to_html_with_images, ignoriamo quelle dall'HTML parsing
        # (che potrebbe trovare i tag <img> che abbiamo aggiunto noi)
        
        # Riorganizza in formato atteso (page -> text)
        pages_text = {}
        for text, page in paragraphs_with_page:
            if page not in pages_text:
                pages_text[page] = []
            pages_text[page].append(text)
        
        # Converti a stringa per pagina
        pages_text_str = {page: "\n\n".join(texts) for page, texts in pages_text.items()}
        
        logger.info("HTML parsato: %d pagine, %d immagini totali", 
                   len(pages_text_str), sum(len(v) for v in images_by_page.values()))
        
    except Exception as e:
        logger.error("Errore nel parsing HTML: %s. Fallback a estrazione diretta.", str(e))
        images_by_page = extract_images(pdf_path)
        captions = load_image_captions()
        return _ingest_pdf_legacy(pdf_path, brand, images_by_page, captions)

    # Continua con il nuovo flusso HTML
    captions = load_image_captions()

    chunks: List[Chunk] = []
    
    for page_index, text in pages_text_str.items():
        logger.info("Chunking pagina %d - testo: %d char", page_index, len(text))
        
        # Estrai i paragrafi mantenendo la struttura
        title_pattern = re.compile(r"^\d+(?:\.\d+)*\.?\s+.+")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        paragraphs: List[str] = []
        current: List[str] = []
        pending_title = ""

        for line in lines:
            if title_pattern.match(line):
                if current:
                    paragraph_text = " ".join(current).strip()
                    if pending_title:
                        paragraph_text = f"{pending_title}\n{paragraph_text}"
                        pending_title = ""
                    paragraphs.append(paragraph_text)
                    current = []
                pending_title = line
                continue

            if pending_title and not current and not title_pattern.match(line):
                if len(line) <= 60 or (line and line[0].islower()):
                    pending_title = f"{pending_title} {line}".strip()
                    continue

            current.append(line)

        if current:
            paragraph_text = " ".join(current).strip()
            if pending_title:
                paragraph_text = f"{pending_title}\n{paragraph_text}"
                pending_title = ""
            paragraphs.append(paragraph_text)
        elif pending_title:
            paragraphs.append(pending_title)

        # Per ogni paragrafo, crea la versione con immagini embedded
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Crea una versione del paragrafo con le immagini incorporate
            full_paragraph_with_images = paragraph
            page_images = images_by_page.get(page_index, [])
            if page_images:
                # Aggiungi le immagini alla fine del paragrafo con le loro didascalie
                image_blocks = []
                for img_url in page_images:
                    caption = captions.get(img_url, "Immagine illustrativa")
                    image_blocks.append(f"[IMMAGINE: {img_url} - {caption}]")
                full_paragraph_with_images = f"{paragraph}\n\n" + "\n".join(image_blocks)

            # Ora chunka il paragrafo se è troppo lungo
            pieces = chunk_text(paragraph, size=1000, overlap=0)
            if not pieces:
                pieces = [paragraph]

            # Per ogni chunk del paragrafo, crea un Chunk con il full_paragraph completo
            for piece in pieces:
                chunk_id = str(uuid.uuid4())
                html_file = ""
                html_anchor = f"chunk-{chunk_id}"
                chunk = Chunk(
                    id=chunk_id,
                    brand=brand,
                    manual=pdf_path.stem,
                    page=page_index,
                    text=piece,  # Il chunk segmentato per la ricerca
                    images=page_images,
                    html_file=html_file,
                    html_anchor=html_anchor,
                    full_paragraph=full_paragraph_with_images,  # Paragrafo completo con immagini per l'LLM
                )
                chunks.append(chunk)
        
        if page_index % 5 == 0 or page_index == 1:
            logger.info("Elaborata pagina %d/%d - %d chunk creati", page_index, len(pages_text), len(chunks))

    # Raggruppa chunk per pagina per l'HTML
    chunks_by_page: Dict[int, List[Chunk]] = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page, []).append(chunk)

    logger.info("Generazione HTML per %s", pdf_path.name)
    html_file = build_html(pdf_path, chunks_by_page, images_by_page)
    for chunk in chunks:
        chunk.html_file = html_file

    logger.info("Creati %s chunk per %s", len(chunks), pdf_path.name)
    return chunks, sum(len(v) for v in images_by_page.values())


def ingest_all() -> IngestResult:
    pdfs = list_manual_pdfs()
    if not pdfs:
        logger.warning("Nessun PDF trovato in manuals/ o root")
        return IngestResult(manuals=0, chunks=0, images=0)

    all_chunks: List[Chunk] = []
    total_images = 0

    # Prima estrai le immagini dai PDF
    for pdf in pdfs:
        chunks, img_count = ingest_pdf(pdf)
        all_chunks.extend(chunks)
        total_images += img_count

    # Rimuovi immagini troppo piccole e pulisci le captions
    # (Disabilitato: vogliamo tutte le immagini)
    # removed = cleanup_small_images()
    # if removed:
    #     logger.info("Rimosse %d immagini troppo piccole", removed)
    
    # DOPO aver estratto le immagini, genera i captions
    logger.info("Generazione didascalie per le immagini...")
    generate_image_captions()
    
    # Ricarica i captions e aggiorna i chunk con le didascalie
    captions = load_image_captions()
    logger.info("Captions caricati: %d immagini", len(captions))

    logger.info("Salvataggio indice con %s chunk", len(all_chunks))
    save_index(all_chunks)
    logger.info("Indice salvato completamente")
    logger.info(
        "Ingestione completata: manuali=%s, chunk=%s, immagini=%s",
        len(pdfs),
        len(all_chunks),
        total_images,
    )
    return IngestResult(manuals=len(pdfs), chunks=len(all_chunks), images=total_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestione manuali PDF")
    parser.add_argument("--reset", action="store_true", help="Cancella il database ChromaDB e reinizializza")
    parser.add_argument("--clean-only", action="store_true", help="Esegue solo la pulizia immagini troppo piccole")
    args = parser.parse_args()

    if args.reset:
        from .rag import CHROMA_DIR
        logger.info("Reset completo: cancellazione database e file generati...")
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info("Directory ChromaDB eliminata")
        images_path = Path(IMAGES_DIR)
        if images_path.exists():
            shutil.rmtree(images_path)
            logger.info("Immagini eliminate")
        html_path = Path(HTML_DIR)
        if html_path.exists():
            shutil.rmtree(html_path)
            logger.info("File HTML eliminati")
        chunks_file = STORAGE_DIR / "chunks.jsonl"
        if chunks_file.exists():
            chunks_file.unlink()
            logger.info("chunks.jsonl eliminato")
        # Non cancellare image_captions.json - riutilizziamo le captions esistenti
        logger.info("Reset completato")
    
    if args.clean_only:
        removed = cleanup_small_images()
        if removed:
            logger.info("Rimosse %d immagini troppo piccole", removed)
        else:
            logger.info("Nessuna immagine da rimuovere")
        return

    ingest_all()


if __name__ == "__main__":
    main()
