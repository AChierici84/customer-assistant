from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
CAPTIONS_FILE = STORAGE_DIR / "image_captions.json"

FALLBACK_CAPTIONS = {
    "Immagine del manuale",
    "Immagine illustrativa",
    "Non disponibile",
}


def load_captions() -> dict[str, str]:
    if CAPTIONS_FILE.exists():
        return json.loads(CAPTIONS_FILE.read_text(encoding="utf-8"))
    return {}


def save_captions(captions: dict[str, str]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONS_FILE.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")


def list_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    return [f"/static/images/{p.name}" for p in sorted(IMAGES_DIR.glob("*.png"))]


def needs_caption(url: str, captions: dict[str, str]) -> bool:
    if url not in captions:
        return True
    value = (captions.get(url) or "").strip()
    if not value:
        return True

    if value in FALLBACK_CAPTIONS:
        return True

    lower_value = value.lower()
    vague_terms = ["non disponibile", "superficie", "gradiente", "manuale"]
    return any(term in lower_value for term in vague_terms)


def prompt_caption(image_url: str) -> str:
    image_path = IMAGES_DIR / Path(image_url).name
    if image_path.exists():
        try:
            os.startfile(image_path)  # Apri con il visualizzatore predefinito (Windows)
        except Exception:
            pass
    print(f"Caption per {Path(image_url).name}:")
    return input("> ").strip()


def main() -> None:
    captions = load_captions()
    images = list_images()

    if not images:
        print("Nessuna immagine trovata in backend/storage/images")
        return

    pending = [img for img in images if needs_caption(img, captions)]

    if not pending:
        print("Tutte le immagini hanno già una caption.")
        return

    print(f"Immagini da completare: {len(pending)}")
    print("Inserisci una didascalia breve (INVIO per saltare).")

    updated = 0
    for image_url in pending:
        caption = prompt_caption(image_url)
        if caption:
            captions[image_url] = caption
            updated += 1
        else:
            if image_url not in captions:
                captions[image_url] = "Non disponibile"

    save_captions(captions)
    print(f"Salvate {updated} nuove didascalie in {CAPTIONS_FILE}")


if __name__ == "__main__":
    main()
