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

import fitz  # PyMuPDF
import pdfplumber

from .rag import Chunk, save_index

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
HTML_DIR = STORAGE_DIR / "html"

BRANDS = {"beko", "electroline", "hisense"}

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


def generate_image_captions() -> Dict[str, str]:
    """Genera didascalie per le immagini usando OpenAI Vision API."""
    from dotenv import load_dotenv
    from openai import OpenAI
    import httpx
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY non configurato, captions non saranno generate")
        return {}
    
    try:
        http_client = httpx.Client(trust_env=False)
        client = OpenAI(api_key=api_key, http_client=http_client)
    except Exception as e:
        logger.error("Errore inizializzazione OpenAI: %s", str(e))
        return {}
    
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
    
    logger.info("Immagini da processare (nuove): %d", len(images_to_process))
    
    if not images_to_process:
        logger.info("Tutte le immagini hanno già una caption, skip generazione")
        return captions
    
    for idx, image_path in enumerate(images_to_process, 1):
        image_url = f"/static/images/{image_path.name}"
        logger.info("Generando caption %d/%d per %s", idx, len(images_to_process), image_path.name)
        
        try:
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Genera una didascalia breve e diretta per questa immagine da manuale tecnico. Usa solo sostantivi e aggettivi essenziali, senza frasi introduttive. Esempi: 'Tabella programmi di lavaggio', 'Schema installazione tubi', 'Carrello estraibile', 'Pannello comandi', 'Diagramma collegamenti elettrici'. Massimo 5 parole."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
            )
            
            caption = response.choices[0].message.content
            captions[image_url] = caption
            logger.info("Caption per %s: %s", image_path.name, caption[:80])
            
            # Delay per evitare rate limit (TPM)
            time.sleep(2.0)
        except Exception as e:
            logger.error("Errore generazione caption per %s: %s", image_path.name, str(e))
            captions[image_url] = "Immagine del manuale"
            time.sleep(3.0)  # Delay più lungo in caso di errore
    
    # Salva captions in JSON
    CAPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS_FILE.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Captions salvate in %s (%d immagini)", CAPTIONS_FILE, len(captions))
    print(f"\n✓ File captions salvato: {CAPTIONS_FILE}")
    
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
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_images: Dict[int, List[str]] = {}

    min_width = 80
    min_height = 80
    min_area = 8000

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        if not image_list:
            continue
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_name = f"{pdf_path.stem}_p{page_index + 1}_{img_index}.png"
            brand = infer_brand(pdf_path.name)
            file_name = f"{brand}_{base_name}"
            image_path = IMAGES_DIR / file_name
            if image_path.exists():
                relative = f"/static/images/{file_name}"
                page_images.setdefault(page_index + 1, []).append(relative)
                continue
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace is None or pix.n > 4 or pix.alpha or pix.n < 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            if pix.width < min_width or pix.height < min_height or (pix.width * pix.height) < min_area:
                logger.debug(
                    "Immagine troppo piccola, salto: %s (%dx%d)",
                    file_name,
                    pix.width,
                    pix.height,
                )
                pix = None
                continue
            try:
                pix.save(image_path.as_posix())
            except ValueError:
                pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                pix_rgb.save(image_path.as_posix())
                pix_rgb = None
            pix = None
            relative = f"/static/images/{file_name}"
            page_images.setdefault(page_index + 1, []).append(relative)
    doc.close()
    logger.info(
        "Estratte %s immagini da %s",
        sum(len(v) for v in page_images.values()),
        pdf_path.name,
    )
    return page_images


def cleanup_small_images(min_width: int = 80, min_height: int = 80, min_area: int = 8000) -> int:
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


def ingest_pdf(pdf_path: Path) -> Tuple[List[Chunk], int]:
    brand = infer_brand(pdf_path.name)
    logger.info("Inizio ingestione manuale %s (brand=%s)", pdf_path.name, brand)
    images_by_page = extract_images(pdf_path)
    captions = load_image_captions()

    chunks: List[Chunk] = []
    pages_text = extract_text_with_structure(pdf_path)
    logger.info("Estratto testo da %s pagine con struttura", len(pages_text))

    for page_index, text in enumerate(pages_text, start=1):
        logger.info("Chunking pagina %d - testo: %d char", page_index, len(text))
        pieces = chunk_text(text)
        logger.info("Pagina %d chunked in %d pieces", page_index, len(pieces))
        if not pieces and images_by_page.get(page_index):
            pieces = [f"Contenuto visivo pagina {page_index}."]

        # Prepara nota sulla disponibilità di immagini per questa pagina CON didascalie
        images_note = ""
        if images_by_page.get(page_index):
            image_descriptions = []
            for img_url in images_by_page[page_index]:
                caption = captions.get(img_url, "Immagine illustrativa")
                image_descriptions.append(f"- {img_url}: {caption}")
            images_note = f"\n\n[Immagini disponibili in questa pagina:\n" + "\n".join(image_descriptions) + "]"

        for piece in pieces:
            chunk_id = str(uuid.uuid4())
            html_file = ""
            html_anchor = f"chunk-{chunk_id}"
            # Includi nota sulle immagini nel testo del chunk
            enhanced_text = piece + images_note if images_note else piece
            chunk = Chunk(
                id=chunk_id,
                brand=brand,
                manual=pdf_path.stem,
                page=page_index,
                text=enhanced_text,
                images=images_by_page.get(page_index, []),
                html_file=html_file,
                html_anchor=html_anchor,
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
    removed = cleanup_small_images()
    if removed:
        logger.info("Rimosse %d immagini troppo piccole", removed)
    
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
