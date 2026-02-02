from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.jsonl")
CHROMA_DIR = os.path.join(STORAGE_DIR, "chroma")
CHROMA_COLLECTION = "manual_chunks"

_MODEL: Optional[SentenceTransformer] = None
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def get_chroma() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
    return _CHROMA_CLIENT


def get_collection() -> chromadb.Collection:
    client = get_chroma()
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


@dataclass
class Chunk:
    id: str
    brand: str
    manual: str
    page: int
    text: str
    images: List[str]
    html_file: str
    html_anchor: str
    full_paragraph: str = ""  # The complete paragraph from source (with images embedded)

    @staticmethod
    def from_json(data: dict) -> "Chunk":
        return Chunk(
            id=data["id"],
            brand=data["brand"],
            manual=data["manual"],
            page=int(data["page"]),
            text=data["text"],
            images=list(data.get("images", [])),
            html_file=data.get("html_file", ""),
            html_anchor=data.get("html_anchor", ""),
            full_paragraph=data.get("full_paragraph", ""),
        )

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "brand": self.brand,
            "manual": self.manual,
            "page": self.page,
            "text": self.text,
            "images": self.images,
            "html_file": self.html_file,
            "html_anchor": self.html_anchor,
            "full_paragraph": self.full_paragraph,
        }


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def save_index(chunks: List[Chunk]) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")

    collection = get_collection()
    try:
        collection.delete(where={})
    except Exception:
        pass

    if not chunks:
        return

    documents = [chunk.text for chunk in chunks]
    embeddings = embed_texts(documents)
    metadatas = [
        {
            "brand": chunk.brand,
            "manual": chunk.manual,
            "page": int(chunk.page),
            "images": json.dumps(chunk.images, ensure_ascii=False),
            "html_file": chunk.html_file,
            "html_anchor": chunk.html_anchor,
        }
        for chunk in chunks
    ]
    ids = [chunk.id for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


def load_index() -> tuple[List[Chunk], np.ndarray]:
    if not os.path.exists(CHUNKS_PATH):
        return [], np.zeros((0, 384), dtype=np.float32)

    chunks: List[Chunk] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_json(json.loads(line)))
    return chunks, np.zeros((0, 384), dtype=np.float32)


def search(
    query: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    top_k: int = 5,
    brand: Optional[str] = None,
) -> List[tuple[Chunk, float]]:
    if not chunks:
        return []

    query_vec = embed_texts([query])[0]
    collection = get_collection()
    where = {"brand": brand.lower()} if brand else None

    response = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    results: List[tuple[Chunk, float]] = []
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    for idx, doc, meta, dist in zip(ids, documents, metadatas, distances):
        images = []
        if meta and "images" in meta and meta["images"]:
            try:
                images = json.loads(meta["images"])
            except Exception:
                images = []
        chunk = Chunk(
            id=idx,
            brand=meta.get("brand", ""),
            manual=meta.get("manual", ""),
            page=int(meta.get("page", 0)),
            text=doc or "",
            images=images,
            html_file=meta.get("html_file", ""),
            html_anchor=meta.get("html_anchor", ""),
        )
        score = 1.0 - float(dist)
        results.append((chunk, score))

    return results
