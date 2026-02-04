#!/usr/bin/env python
"""Analizza contenuto grafico dei PDF per distinguere raster vs vettoriali."""

import fitz
from pathlib import Path

def analyze_pdf(pdf_path: Path):
    """Analizza un PDF e mostra dettagli su immagini raster e contenuto vettoriale."""
    doc = fitz.open(pdf_path)
    
    print(f"\n{'='*70}")
    print(f"📄 Analisi: {pdf_path.name}")
    print(f"{'='*70}")
    print(f"Pagine totali: {len(doc)}\n")
    
    total_raster = 0
    total_drawings = 0
    
    for page_num in range(min(5, len(doc))):  # Analizza prime 5 pagine
        page = doc[page_num]
        
        # Immagini raster
        images = page.get_images(full=True)
        
        # Elementi vettoriali (drawings)
        drawings = page.get_drawings()
        
        if images or drawings:
            print(f"📑 Pagina {page_num + 1}:")
            print(f"  🖼️  Immagini raster: {len(images)}")
            
            for img_idx, img in enumerate(images[:3], 1):  # Prime 3 immagini
                xref = img[0]
                try:
                    img_info = doc.extract_image(xref)
                    print(f"      #{img_idx}: {img_info['width']}x{img_info['height']}px, "
                          f"formato={img_info['ext']}, {len(img_info['image'])/1024:.1f}KB")
                except:
                    print(f"      #{img_idx}: xref={xref} (errore estrazione)")
            
            print(f"  🎨 Elementi vettoriali: {len(drawings)}")
            
            if drawings:
                # Classifica tipi di elementi
                paths = sum(1 for d in drawings if d.get('type') == 'path')
                rects = sum(1 for d in drawings if d.get('type') in ['re', 'rect'])
                fills = sum(1 for d in drawings if d.get('fill'))
                strokes = sum(1 for d in drawings if d.get('stroke'))
                
                print(f"      - Path: {paths}, Rettangoli: {rects}")
                print(f"      - Con riempimento: {fills}, Con bordo: {strokes}")
            
            print()
        
        total_raster += len(images)
        total_drawings += len(drawings)
    
    print(f"{'='*70}")
    print(f"📊 TOTALE (prime {min(5, len(doc))} pagine):")
    print(f"  🖼️  Immagini raster: {total_raster}")
    print(f"  🎨 Elementi vettoriali: {total_drawings}")
    print(f"{'='*70}\n")
    
    doc.close()
    
    # Suggerimenti
    if total_raster > 0 and total_drawings > 100:
        print("💡 Questo PDF contiene ENTRAMBI:")
        print("   - Immagini raster (foto, screenshot) → estraibili con get_images()")
        print("   - Molti elementi vettoriali (diagrammi, grafici) → servono drawings o rendering pagina\n")
    elif total_raster > 0 and total_drawings < 50:
        print("💡 Questo PDF è prevalentemente RASTER:")
        print("   - Le immagini sono incorporate come file → get_images() funzionerà bene\n")
    elif total_raster == 0 and total_drawings > 0:
        print("💡 Questo PDF è SOLO VETTORIALE:")
        print("   - Grafici/diagrammi disegnati come path → serve pdf2image o rendering pagina\n")
    else:
        print("💡 PDF prevalentemente testuale con pochi elementi grafici\n")


if __name__ == "__main__":
    manuals_dir = Path("manuals")
    pdfs = list(manuals_dir.glob("*.pdf"))
    
    if not pdfs:
        print("❌ Nessun PDF trovato in manuals/")
    else:
        print(f"\n🔍 Trovati {len(pdfs)} PDF da analizzare\n")
        for pdf in pdfs[:3]:  # Analizza primi 3 PDF
            analyze_pdf(pdf)
