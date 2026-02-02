from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from .ingest import IngestResult, ingest_all
from .rag import Chunk, load_index, search

app = FastAPI(title="Customer Assistant RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")


class QueryRequest(BaseModel):
    question: str
    brand: Optional[str] = None
    top_k: int = 5


class SourceItem(BaseModel):
    chunk_id: str
    brand: str
    manual: str
    page: int
    score: float
    link: str


class ImageItem(BaseModel):
    chunk_id: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    images: List[ImageItem]


class IngestResponse(BaseModel):
    manuals: int
    chunks: int
    images: int


_cached_chunks: List[Chunk] = []


def load_cache() -> None:
    global _cached_chunks
    _cached_chunks, _ = load_index()


def build_llm_answer(question: str, context: List[str], images: List[ImageItem]) -> str:
    if not OPENAI_API_KEY:
        return "\n\n".join(context) if context else "Nessun risultato trovato."

    try:
        import httpx
        # Disabilita proxies automatici per evitare conflitti
        http_client = httpx.Client(trust_env=False)
        if OPENAI_BASE_URL:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, http_client=http_client)
        else:
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
    except Exception as e:
        return f"Errore nella inizializzazione LLM: {str(e)}"
    
    # Carica le captions delle immagini
    import json
    from pathlib import Path
    captions_file = Path(__file__).parent / "storage" / "image_captions.json"
    captions = {}
    if captions_file.exists():
        try:
            captions = json.loads(captions_file.read_text(encoding="utf-8"))
        except:
            pass
    
    # Crea lista immagini con captions
    image_list_items = []
    for img in images:
        caption = captions.get(img.url, "Immagine illustrativa")
        image_list_items.append(f"- {img.url} ({caption})")
    image_list = "\n".join(image_list_items) or "(nessuna)"
    context_text = "\n\n".join(context) if context else "(nessun contenuto)"

    system_prompt = (
        "Sei un assistente tecnico. Rispondi in italiano in modo chiaro, "
        "preciso e con elenco puntato dei passi operativi. "
        "Usa solo le informazioni fornite nel contesto. "
        "IMPORTANTE: Quando menzioni tabelle, diagrammi o riferimenti visivi, "
        "includi SEMPRE nella risposta i link alle immagini usando il formato markdown: "
        "[DIDASCALIA_BREVE](URL). Usa come didascalia quella fornita tra parentesi per ogni immagine. "
        "Le immagini verranno visualizzate inline con la didascalia sotto."
    )
    user_prompt = (
        f"Domanda:\n{question}\n\n"
        f"Contesto (chunk):\n{context_text}\n\n"
        f"Immagini disponibili (URL e didascalia):\n{image_list}\n\n"
        "Genera una risposta dettagliata. Quando fai riferimento a tabelle o diagrammi, "
        "inserisci il link markdown usando ESATTAMENTE l'URL e la didascalia fornita, "
        "ad esempio: [Tabella dei programmi di lavaggio](/static/images/beko_p23_1.png)"
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        # Normalizza gli URL delle immagini: converte domini casuali a path relativi
        answer = normalize_image_urls(answer, images)
        return answer
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Errore nella generazione della risposta: {str(e)}\n\nDettagli:\n{error_detail}"


def normalize_image_urls(text: str, images: List[ImageItem]) -> str:
    """Corregge gli URL delle immagini nel testo dell'LLM, convertendo domini a path relativi."""
    import re
    
    # Estrai solo i nomi file dagli URL
    image_names = {img.url.split('/')[-1]: img.url for img in images}
    
    # Trova tutti i link markdown [testo](url) 
    def replace_url(match):
        prefix = match.group(1)  # [testo]
        url = match.group(2)     # url
        
        # Estrai il nome file dall'URL (ultimo segmento dopo l'ultimo /)
        filename = url.split('/')[-1]
        
        # Se il nome file è in images, usa l'URL corretto
        if filename in image_names:
            return f"{prefix}({image_names[filename]})"
        
        # Se l'URL non inizia con /, convertilo a path relativo
        if not url.startswith('/') and filename in image_names:
            return f"{prefix}({image_names[filename]})"
        
        return match.group(0)
    
    # Pattern: [testo](url)
    text = re.sub(r'(\[[^\]]+\])\(([^)]+)\)', replace_url, text)
    return text


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    result: IngestResult = ingest_all()
    load_cache()
    return IngestResponse(manuals=result.manuals, chunks=result.chunks, images=result.images)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    if not _cached_chunks:
        load_cache()

    results = search(
        query=payload.question,
        chunks=_cached_chunks,
        embeddings=None,
        top_k=payload.top_k,
        brand=payload.brand,
    )

    sources: List[SourceItem] = []
    images: List[ImageItem] = []
    context_parts: List[str] = []

    for chunk, score in results:
        link = f"{chunk.html_file}#{chunk.html_anchor}" if chunk.html_file else ""
        sources.append(
            SourceItem(
                chunk_id=chunk.id,
                brand=chunk.brand,
                manual=chunk.manual,
                page=chunk.page,
                score=score,
                link=link,
            )
        )
        context_parts.append(chunk.text)
        for img in chunk.images:
            images.append(ImageItem(chunk_id=chunk.id, url=img))

    answer = build_llm_answer(payload.question, context_parts, images)

    return QueryResponse(answer=answer, sources=sources, images=images)
