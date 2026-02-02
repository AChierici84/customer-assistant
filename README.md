# Customer Assistant - Sistema RAG per Manuali Tecnici

Sistema di assistenza clienti basato su RAG (Retrieval-Augmented Generation) per interrogare manuali tecnici di elettrodomestici con supporto per immagini e didascalie.

## Caratteristiche

- **Estrazione PDF Intelligente**: Estrae testo e immagini da manuali PDF con riconoscimento della struttura (titoli, paragrafi)
- **Generazione Automatica Didascalie**: Utilizza OpenAI Vision API per generare descrizioni delle immagini
- **Ricerca Semantica**: ChromaDB + sentence-transformers per ricerca vettoriale nei manuali
- **Contesto Completo per LLM**: Le immagini sono embedded nei paragrafi e il LLM riceve l'intero paragrafo originale (non solo i chunk)
- **Interfaccia Web**: Frontend React con visualizzazione di risposte, sorgenti e immagini
- **HTML Navigabile**: Genera documenti HTML stilizzati per ogni manuale con navigazione per chunk

## Architettura

### Backend (FastAPI + Python)

```
backend/
├── app.py              # API endpoints (query, ingest, health)
├── ingest.py           # Pipeline di ingestione PDF
├── rag.py              # Gestione chunking e vettorizzazione
├── caption_review.py   # Tool per revisione manuale didascalie
├── requirements.txt    # Dipendenze Python
└── storage/
    ├── chunks.jsonl           # Chunk indicizzati
    ├── image_captions.json    # Didascalie immagini
    ├── images/                # Immagini estratte
    ├── html/                  # Documenti HTML generati
    └── chroma/                # Database vettoriale ChromaDB
```

### Frontend (React + Vite)

```
frontend/
├── src/
│   ├── App.jsx         # Componente principale
│   └── main.jsx        # Entry point
├── index.html
├── package.json
└── vite.config.js
```

### Batch Scripts

- `ingest.bat` - Esegue ingestione completa dei PDF
- `start-backend.bat` - Avvia server FastAPI
- `start-frontend.bat` - Avvia dev server Vite
- `caption-review.bat` - Tool interattivo per revisione didascalie

## Installazione

### Prerequisiti

- Python 3.11+
- Node.js 18+
- OpenAI API Key (per generazione didascalie e risposte)

### Setup Backend

```bash
# Crea virtual environment
python -m venv .venv

# Attiva virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installa dipendenze
pip install -r backend/requirements.txt
```

### Setup Frontend

```bash
cd frontend
npm install
```

### Configurazione

Crea un file `.env` nella root del progetto:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Utilizzo

### 1. Preparazione Manuali

Inserisci i file PDF nella cartella `manuals/` (o nella root del progetto). Il sistema riconosce automaticamente i brand: `beko`, `electroline`, `hisense`.

### 2. Ingestione

Esegui l'ingestione completa:

```bash
.\ingest.bat
```

Opzioni disponibili:

```bash
# Reset completo (cancella database e reinizializza)
python -m backend.ingest --reset

# Solo pulizia immagini piccole
python -m backend.ingest --clean-only
```

**Processo di ingestione:**

1. Estrazione immagini dai PDF (filtraggio dimensioni min: 80x80, area min: 8000px)
2. Estrazione testo con preservazione struttura (titoli, paragrafi)
3. Segmentazione in chunk (max 1000 caratteri, no overlap)
4. Embedding immagini nei paragrafi con formato `[IMMAGINE: URL - DIDASCALIA]`
5. Generazione didascalie con OpenAI Vision (con rate limiting: 2s delay)
6. Creazione HTML navigabili per ogni manuale
7. Indicizzazione vettoriale con ChromaDB

### 3. Revisione Didascalie (Opzionale)

Per migliorare le didascalie generate automaticamente:

```bash
.\caption-review.bat
```

Il tool:
- Identifica didascalie vaghe o mancanti
- Mostra le immagini nel visualizzatore di sistema
- Permette modifica interattiva delle didascalie
- Salva automaticamente in `image_captions.json`

### 4. Avvio Server

**Backend:**
```bash
.\start-backend.bat
# Server disponibile su http://localhost:8000
```

**Frontend:**
```bash
.\start-frontend.bat
# App disponibile su http://localhost:5173
```

### 5. Utilizzo Interfaccia Web

1. Apri http://localhost:5173
2. Seleziona brand (opzionale)
3. Inserisci domanda (es: "Come pulire il filtro?")
4. Visualizza:
   - Risposta generata dal LLM
   - Immagini referenziate con didascalie
   - Sorgenti (chunk) con link ai documenti HTML

## API Endpoints

### `POST /query`

Esegue ricerca semantica e genera risposta con LLM.

**Request:**
```json
{
  "question": "Come impostare il programma cotone?",
  "top_k": 5,
  "brand": "beko"  // opzionale
}
```

**Response:**
```json
{
  "answer": "Per impostare il programma cotone...",
  "sources": [
    {
      "chunk_id": "uuid",
      "brand": "beko",
      "manual": "beko",
      "page": 23,
      "score": 0.85,
      "link": "/static/html/beko.html#chunk-uuid"
    }
  ],
  "images": [
    {
      "chunk_id": "uuid",
      "url": "/static/images/beko_p23_1.png"
    }
  ]
}
```

### `POST /ingest`

Esegue ingestione completa dei PDF.

**Response:**
```json
{
  "manuals": 3,
  "chunks": 352,
  "images": 87
}
```

### `GET /health`

Verifica stato del server.

## Architettura Chunk e Paragrafi

### Struttura Chunk

Ogni chunk contiene:

- `text`: Porzione del paragrafo usata per ricerca semantica (max 1000 caratteri)
- `full_paragraph`: Paragrafo completo originale con immagini embedded (usato per LLM)
- `images`: Lista URL immagini associate alla pagina
- `page`, `brand`, `manual`: Metadati
- `html_file`, `html_anchor`: Link al documento HTML

**Esempio:**

```json
{
  "id": "uuid",
  "text": "2. Istruzioni importanti per l'ambiente Questo apparecchio è conforme alla Direttiva WEEE...",
  "full_paragraph": "2. Istruzioni importanti per l'ambiente Questo apparecchio è conforme alla Direttiva WEEE...\n\n[IMMAGINE: /static/images/beko_p5_1.png - Simbolo riciclaggio WEEE]",
  "images": ["/static/images/beko_p5_1.png"],
  "page": 5,
  "brand": "beko",
  "manual": "beko"
}
```

### Flusso Query → LLM

1. **Ricerca Semantica**: L'utente fa una domanda → il sistema cerca nei `text` dei chunk (segmenti brevi)
2. **Recupero Contesto Completo**: Per ogni chunk trovato, recupera il `full_paragraph` (paragrafo completo)
3. **Invio a LLM**: Il LLM riceve:
   - I paragrafi completi (non solo i chunk)
   - Le immagini embedded nel testo con formato `[IMMAGINE: URL - DIDASCALIA]`
   - La lista delle immagini disponibili con didascalie
4. **Generazione Risposta**: L'LLM genera una risposta usando il contesto completo e inserisce link markdown alle immagini

**Vantaggi:**
- ✅ LLM vede sempre il contesto completo (non tronco)
- ✅ Immagini posizionate nel loro paragrafo originale
- ✅ Ricerca semantica efficiente su chunk piccoli
- ✅ Didascalie aiutano l'LLM a capire cosa mostrano le immagini

## Dipendenze Principali

### Backend
- `fastapi` - Framework API
- `chromadb` - Database vettoriale
- `sentence-transformers` - Embedding per ricerca semantica
- `pymupdf` (fitz) - Estrazione immagini da PDF
- `pdfplumber` - Estrazione testo da PDF
- `openai` - API per Vision e LLM
- `httpx` - HTTP client per OpenAI

### Frontend
- `react` - UI framework
- `vite` - Build tool
- `react-markdown` - Rendering markdown nelle risposte

## Funzionalità Avanzate

### Filtraggio Immagini

Il sistema filtra automaticamente:
- Immagini < 80x80 pixel
- Immagini con area < 8000 pixel quadrati
- Icone, loghi, elementi grafici decorativi

### Chunking Intelligente

- Riconoscimento titoli numerati (es: "1.2.3 Installazione")
- Preservazione relazione titolo-contenuto
- Handling titoli spezzati su più righe
- Separazione paragrafi senza overlap

### Rate Limiting OpenAI

- Delay di 2 secondi tra richieste Vision
- Delay di 3 secondi dopo errori
- Cache delle didascalie esistenti (non rigenerate)

### HTML Generati

I documenti HTML includono:
- Indice navigabile
- Chunk stilizzati con bordi e header
- Gallery di immagini con didascalie
- Link diretti ai chunk specifici
- Rendering formattato di titoli ed elenchi puntati

## 🐛 Troubleshooting

### Errore "No module named 'fitz'"
```bash
pip install PyMuPDF
```

### Errore rate limit OpenAI
Le didascalie vengono generate con delay automatico. Se persistente, aumenta il delay in `ingest.py:238`:
```python
time.sleep(3.0)  # Aumenta il delay
```

### ChromaDB telemetry errors
Errori benigni, non influenzano il funzionamento. Possono essere ignorati.

### Immagini non visualizzate
Verifica che i path relativi siano corretti. Il backend serve:
- `/static/images/*` → `backend/storage/images/`
- `/static/html/*` → `backend/storage/html/`

## Note di Sviluppo

### Aggiunta Nuovi Brand

Modifica `BRANDS` in `backend/ingest.py`:
```python
BRANDS = {"beko", "electroline", "hisense", "nuovo_brand"}
```

### Personalizzazione Prompt LLM

Modifica `system_prompt` e `user_prompt` in `backend/app.py:build_llm_answer()`.

### Modifica Dimensione Chunk

Modifica `chunk_text(text, size=1000, overlap=0)` in `backend/ingest.py`.

## Licenza

Progetto interno per assistenza clienti.

## Contributi

Per domande o problemi, contatta il team di sviluppo.

---

**Ultimo aggiornamento**: Febbraio 2026  
**Versione**: 2.0 (con supporto paragrafi completi e immagini embedded)
