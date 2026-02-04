# Customer Assistant - Sistema RAG Multimodale per Manuali Tecnici

Sistema di assistenza clienti basato su RAG (Retrieval-Augmented Generation) per interrogare manuali tecnici di elettrodomestici con supporto per immagini e didascalie.

## Caratteristiche

- **Estrazione PDF Intelligente (Raster + Vettoriale)**: 
  - Immagini raster (foto, screenshot) con `get_images()`
  - Elementi vettoriali (diagrammi, grafici) con `get_drawings()` + rendering
  - Clustering automatico dei disegni per estrarre grafici significativi
- **Ricerca Semantica**: ChromaDB + sentence-transformers per ricerca vettoriale nei manuali
- **Contesto Completo per LLM**: Le immagini sono embed nei paragrafi e il LLM riceve l'intero paragrafo originale (non solo i chunk)
- **Upload File Web**: Carica PDF direttamente dall'interfaccia, ingestione automatica
- **Interfaccia Web**: Frontend React con visualizzazione di risposte, sorgenti e immagini
- **HTML Navigabile**: Genera documenti HTML per ogni manuale con navigazione per pagina/chunk
- **Didascalie Manuali**: Tool interattivo (caption-review.bat) per revisione e editing manuale delle didascalie
- **Supporto Multi-Marca**: aspirapolvere, condizionatore, congelatore, lavastoviglie, lavatrice, microonde, plasma

## Architettura

### Backend (FastAPI + Python)

```
backend/
├── app.py              # API endpoints (query, upload, health)
├── ingest.py           # Pipeline di ingestione PDF (PDF→HTML→Chunking)
├── rag.py              # Gestione chunking e vettorizzazione
├── caption_review.py   # Tool per revisione manuale didascalie
├── requirements.txt    # Dipendenze Python
└── storage/
    ├── chunks.jsonl           # Chunk indicizzati
    ├── image_captions.json    # Didascalie immagini (manuali)
    ├── images/                # Immagini estratte dai PDF
    ├── html/                  # Documenti HTML generati da PDF
    └── chroma/                # Database vettoriale ChromaDB
```

**Pipeline di Ingestione:**
1. `pdf_to_html_with_images()` - Converte PDF a HTML usando PyMuPDF (fitz), estrae immagini raster e vettoriali
   - Testo: `page.get_text("html")` per HTML strutturato
   - Immagini Raster: `get_images(full=True)` per foto e screenshot
   - Elementi Vettoriali: `get_drawings()` per diagrammi e grafici (con clustering automatico)
2. `parse_html_for_content()` - Estrae testo dall'HTML, organizza per pagina
3. `chunk_text()` - Chunka il testo mantenendo la struttura (512 token per chunk)
4. Embedding e Indicizzazione - ChromaDB indicizza i chunk per ricerca semantica

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

- `start-backend.bat` - Avvia server FastAPI
- `start-frontend.bat` - Avvia dev server Vite
- `ingest.bat` - Esegue ingestione completa dei PDF
- `caption-review.bat` - Tool interattivo per revisione didascalie

## Installazione

### Prerequisiti

- Python 3.11+
- Node.js 18+

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

Crea un file `.env` nella root del progetto (copia da `.env.example`):

```env
# OpenAI Configuration (opzionale, solo per risposte LLM)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
```

Le didascalie delle immagini sono gestite manualmente tramite il tool `caption-review.bat` (vedi sezione dedicata).

## Utilizzo

### 1. Avvio Server

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

### 2. Upload Manuali (Metodo Consigliato)

1. Apri http://localhost:5173
2. Clicca su **"📤 Carica manuale"**
3. Seleziona un file PDF
4. Sistema salva in `manuals/` e ingestisce automaticamente
5. Ricevi conferma con numero chunk e immagini

### 3. Ingestione da Linea di Comando

Inserisci i file PDF nella cartella `manuals/` e esegui:

```bash
python -m backend.ingest --reset  # Reset completo
python -m backend.ingest           # Ingestione incrementale
```

**Nuovo processo di ingestione (PDF→HTML):**

1. **Conversione PDF→HTML**: PyMuPDF (fitz) estrae testo strutturato con `page.get_text("html")`
2. **Estrazione Immagini Raster**: `get_images(full=True)` estrae foto e screenshot direttamente dai flussi PDF
3. **Estrazione Elementi Vettoriali**: `get_drawings()` identifica diagrammi, grafici e forme vettoriali
   - Clustering automatico per raggruppare elementi correlati (espansione ±5px)
   - Filtering (minimo 50px, area minima 2500px²) per eliminare decorazioni
   - Rendering pixmap per convertire vettoriali in PNG
4. **Parsing HTML**: BeautifulSoup estrae paragrafi mantenendo il contesto per pagina
5. **Chunking**: Testo segmentato in chunk (max 512 token) rispettando i paragrafi
6. **Embedding Immagini**: Immagini inserite nei paragrafi con formato `[IMMAGINE: URL]`
7. **Creazione HTML**: Genera documento HTML navigabile per ogni manuale
8. **Indicizzazione**: ChromaDB indicizza i chunk per ricerca vettoriale
9. **Gestione Didascalie**: Caricamento didascalie da `image_captions.json`

### 4. Revisione Didascalie (Opzionale)

Per aggiungere o modificare didascalie manualmente:

```bash
.\caption-review.bat
```

Il tool:
- Mostra immagini estratte dal PDF
- Permette di aggiungere o modificare didascalie
- Salva automaticamente in `image_captions.json`
- Usato dal sistema nella generazione delle risposte

### 5. Utilizzo Interfaccia Web

1. Apri http://localhost:5173
2. (Opzionale) Seleziona marca per filtrare risultati
3. **📤 Carica manuale** - Upload file PDF per ingestione automatica
4. Scrivi la domanda (es: "Come pulire il filtro?")
5. Visualizza:
   - Risposta generata dal LLM con immagini inline
   - Numero messaggi nella chat
   - Pulsante "Svuota chat" per resettare la conversazione
   - Sorgenti (chunk) con link ai documenti HTML
   - Immagini correlate estratte dai manuali

Esempi di ricerche:

Risposta con markdown costruita a partire dal chunk:
<img width="1834" height="798" alt="Screenshot 2026-02-04 082429" src="https://github.com/user-attachments/assets/6492e82d-7def-457b-8a0f-63bbcaabb674" />

Gestione delle immagine inline nella risposta.
<img width="1677" height="848" alt="Screenshot 2026-02-04 082443" src="https://github.com/user-attachments/assets/87d098c3-d91b-41d3-b830-3ae41110cd28" />

Fonti e immagini associate:
<img width="1325" height="633" alt="Screenshot 2026-02-04 082501" src="https://github.com/user-attachments/assets/d3319e74-2a5c-400a-bd35-927c55215d7e" />

Cliccando si può visualizzare la versione html del pdf coi chunk (Massima trasparenza sul comportamento della rag ed explainability)
<img width="1589" height="728" alt="Screenshot 2026-02-04 082604" src="https://github.com/user-attachments/assets/2de4a060-fab6-4e91-933a-3a57fefbc192" />

L'LLM è istruito da prompt per fornire passi chiari con elenchi e guidare l'utente nelle sue necessità.
<img width="1168" height="758" alt="Screenshot 2026-02-04 082744" src="https://github.com/user-attachments/assets/143c9055-7f60-4d1b-bf8e-8456cd3cef18" />
Allegando le immagini che ritiene necessarie. La caption in questo senso aiuta l'llm nel comprendere che cosa è rappresentato.
<img width="1292" height="643" alt="Screenshot 2026-02-04 082759" src="https://github.com/user-attachments/assets/22951685-dafa-4b6b-9110-7d06755d4ef5" />

Esempi aggiuntivi con prodotti differenti:
<img width="1624" height="866" alt="Screenshot 2026-02-04 083043" src="https://github.com/user-attachments/assets/3e31f1cc-3a34-44d9-9f26-dddab1c2c08c" />

Lavastoviglie...
<img width="1056" height="657" alt="Screenshot 2026-02-04 083240" src="https://github.com/user-attachments/assets/3d486082-f941-4575-be2b-4410ed880060" />

Lavatrice con guida all'installazione.
<img width="1092" height="865" alt="Screenshot 2026-02-04 083535" src="https://github.com/user-attachments/assets/771e989f-2798-41dc-b1d1-c36f3dc9e03e" />
<img width="887" height="580" alt="Screenshot 2026-02-04 083542" src="https://github.com/user-attachments/assets/958743e7-2342-4ec0-875f-46f27ad1e7c2" />

Microonde con consigli d'uso.
<img width="1185" height="827" alt="Screenshot 2026-02-04 083729" src="https://github.com/user-attachments/assets/30c1a094-4cc9-45e5-a34a-31473b734a42" />

Televisore al plasma
<img width="1144" height="834" alt="Screenshot 2026-02-04 084111" src="https://github.com/user-attachments/assets/d9ec3665-1b55-4ca6-b23e-d0a7cae448b0" />


Non ci sono personalizzazione per manuali specifici, può essere convertito qualsiasi manuale in formato pdf.

## API Endpoints

### `POST /upload`

Carica un file PDF, lo salva in `manuals/` ed esegue l'ingestione.

**Request:**
```
Content-Type: multipart/form-data
file: [binary PDF file]
```

**Response:**
```json
{
  "manuals": 1,
  "chunks": 156,
  "images": 42
}
```

### `POST /query`

Esegue ricerca semantica e genera risposta con LLM.

**Request:**
```json
{
  "question": "Come impostare il programma cotone?",
  "top_k": 5,
  "brand": null  // opzionale, null per tutte le marche
}
```

**Response:**
```json
{
  "answer": "Per impostare il programma cotone...",
  "sources": [
    {
      "chunk_id": "uuid",
      "brand": "aspirapolvere",
      "manual": "aspirapolvere",
      "page": 12,
      "score": 0.87,
      "link": "/static/html/aspirapolvere.html#chunk-uuid"
    }
  ],
  "images": [
    {
      "chunk_id": "uuid",
      "url": "/static/images/aspirapolvere_p12_1.png"
    }
  ]
}
```

### `GET /health`

Verifica stato del server.

**Response:**
```json
{
  "status": "ok"
}
```

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
  "full_paragraph": "2. Istruzioni importanti per l'ambiente Questo apparecchio è conforme alla Direttiva WEEE...\n\n[IMMAGINE: /static/images/aspirapolvere_p5_1.png - Simbolo riciclaggio WEEE]",
  "images": ["/static/images/aspirapolvere_p5_1.png"],
  "page": 5,
  "brand": "aspirapolvere",
  "manual": "aspirapolvere"
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
- `pymupdf` (fitz) - Estrazione testo e immagini da PDF
- `pdfplumber` - Estrazione testo strutturato da PDF (fallback legacy)
- `pdf2image` - Conversione pagine PDF a immagini (fallback legacy)
- `beautifulsoup4` - Parsing HTML dai PDF convertiti
- `python-multipart` - Supporto file upload multipart
- `openai` - API per LLM (solo risposte, non per immagini)
- `httpx` - HTTP client per OpenAI

### Frontend
- `react` - UI framework
- `vite` - Build tool
- `react-markdown` - Rendering markdown nelle risposte
- `@mui/material` - Componenti UI Material Design

## Funzionalità Avanzate

### Estrazione Elementi Vettoriali (Funzione: `extract_vector_graphics_as_images()`)

I manuali contengono spesso diagrammi, schemi e grafici in formato vettoriale (non raster). La funzione **`extract_vector_graphics_as_images()`** estrae questi elementi:

**Algoritmo:**
1. **Identificazione**: `page.get_drawings()` ottiene tutti gli elementi vettoriali (rette, poligoni, curve, testo vettoriale)
2. **Clustering**: Raggruppa disegni vicini (espansione ±5px dei bounding box) per identificare forme logiche
3. **Filtering**: Elimina elementi piccoli (< 50px) o con area < 2500px² (decorazioni)
4. **Rendering**: Converte ogni cluster in pixmap PNG a zoom 1.0x per preservare qualità
5. **Salvataggio**: File PNG con naming `{brand}_{filename}_p{page}_vec{idx}.png`

**Esempio output:**
- Input: PDF con diagramma elettrico vettoriale (392 elementi identificati)
- Output: PNG separate per ogni schema/diagramma (es: `aspirapolvere_manual_p8_vec0.png`, `aspirapolvere_manual_p8_vec1.png`)

**Codice:**
```python
def extract_vector_graphics_as_images(
    page,
    brand: str,
    stem: str,
    page_num: int,
    output_dir: str,
    min_size: int = 50,
    min_area: int = 2500
) -> List[str]:
    """
    Estrae elementi vettoriali da pagina PDF e li converte in PNG.
    
    Args:
        page: Pagina PyMuPDF
        brand: Nome marca (per naming file)
        stem: Filename base
        page_num: Numero pagina
        output_dir: Directory di output
        min_size: Dimensione minima elemento (pixel)
        min_area: Area minima elemento (pixel²)
    
    Returns:
        Lista di percorsi immagini estratte
    """
```

**Vantaggi:**
- ✅ Diagrammi e schemi estratti come immagini separate
- ✅ Preserva qualità vettoriale con rendering a zoom 1.0x
- ✅ Clustering automatico raggruppa elementi correlati
- ✅ Filtra automaticamente decorazioni e elementi piccoli
- ✅ Fallback per elementi raster se rendering vettoriale fallisce

### Pipeline PDF→HTML

La nuova pipeline sfrutta PyMuPDF per:
- **Estrazione testo strutturato**: `page.get_text("html")` preserva il layout
- **Immagini raster**: `get_images(full=True)` estrae foto e screenshot
- **Elementi vettoriali**: `extract_vector_graphics_as_images()` estrae diagrammi
- **Parsing HTML con BeautifulSoup**: Organizza il contenuto per pagina
- **Fallback intelligente**: Se estrazione fallisce, usa metodo legacy

### Filtraggio e Cleaning

Automatico durante parsing:
- Ignora testo troppo corto (< 3 caratteri)
- Preserva struttura paragrafi
- Associa immagini alle pagine corrette
- Valida dimensioni clip_rect (minimo 10x10px)

### Chunking Intelligente

- Segmenti massimi 512 token
- Preservazione relazione titolo-contenuto
- Separazione paragrafi logici
- Embedding tramite sentence-transformers (all-MiniLM-L6-v2)

### HTML Generati

I documenti HTML includono:
- Indice navigabile per pagina
- Testo strutturato preservando layout
- Gallery di immagini (raster + vettoriali) con didascalie
- Link diretti ai chunk specifici
- Rendering pulito per la lettura

### Gestione Didascalie

- **Sorgente**: `image_captions.json` (manuale)
- **Tool di revisione**: `caption-review.bat`
- **Formato storage**: `{"/static/images/brand_filename.png": "descrizione"}`
- **Utilizzo**: Embeddato nei paragrafi nel formato `[IMMAGINE: URL - DIDASCALIA]`

**A differenza di versioni precedenti**, non c'è generazione automatica di didascalie (BLIP rimosso). 
Le didascalie sono gestite manualmente tramite il tool di revisione per garantire qualità migliore.

### UI Frontend

- ✅ Caricamento file PDF drag-and-drop
- ✅ Status di caricamento in tempo reale
- ✅ Contatore messaggi chat
- ✅ Pulsante "Svuota chat" per reset conversazione
- ✅ Filtro marche (tutte/aspirapolvere/condizionatore/...)
- ✅ Rendering markdown delle risposte
- ✅ Visualizzazione immagini inline
- ✅ Link clicabili alle fonti

## 🐛 Troubleshooting

### Errore "No HTML output from pdfplumber"
Questo significa che pdfplumber non ha estratto il testo dal PDF (raro). Fallback automatico a estrazione legacy.

### ChromaDB telemetry errors
Errori benigni, non influenzano il funzionamento. Possono essere ignorati.

### Immagini non visualizzate
Verifica che i path relativi siano corretti. Il backend serve:
- `/static/images/*` → `backend/storage/images/`
- `/static/html/*` → `backend/storage/html/`

## Note di Sviluppo

### Aggiunta Nuovi Brand

I brand vengono inferiti dai nomi file. Per aggiungere un brand:

1. Modifica `BRANDS` in `backend/ingest.py`:
```python
BRANDS = {"aspirapolvere", "condizionatore", "congelatore", "lavastoviglie", "lavatrice", "microonde", "plasma", "nuovo_brand"}
```

2. (Opzionale) Aggiorna `brandOptions` in `frontend/src/App.jsx` per il filtro:
```javascript
const brandOptions = ["", "aspirapolvere", "condizionatore", "congelatore", "lavastoviglie", "lavatrice", "microonde", "plasma", "nuovo_brand"];
```

### Personalizzazione Prompt LLM

Modifica `system_prompt` e `user_prompt` in `backend/app.py:build_llm_answer()`.

### Modifica Dimensione Chunk

Modifica `chunk_text(text, size=1000, overlap=0)` in `backend/ingest.py:chunk_text()`.

## Licenza

Progetto interno per assistenza clienti.

## Contributi

Per domande o problemi, contatta il team di sviluppo.

---

**Ultimo aggiornamento**: Febbraio 2026  
**Versione**: 3.0 (PDF→HTML pipeline, didascalie manuali)
