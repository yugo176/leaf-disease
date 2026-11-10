"""API HTTP de diagnostic des feuilles.

Lancer :  uvicorn api.main:app --port 8000
Docs interactives : http://localhost:8000/docs

POST /predict  (multipart, champ « file » = une image ; paramètre « crop » facultatif)
→ les 3 diagnostics les plus probables, avec probabilités et conseils.
La route est une fonction `def` : FastAPI l'exécute dans un thread, le serveur reste disponible.
L'image est lue en mémoire et n'est jamais enregistrée.
"""

import io
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from leaf_core.labels import CROPS, GENERAL_WARNING
from leaf_core.model import LOW_CONFIDENCE, Predictor

MAX_BYTES = 10 * 1024 * 1024
predictor = Predictor(Path(__file__).resolve().parent.parent / "model")

app = FastAPI(title="Maladies des feuilles", version="1.0.0")
origins = [o for o in os.getenv("LEAF_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta["arch"]}


@app.get("/classes")
def classes():
    from leaf_core.labels import CLASSES

    return [{"id": c.id, "crop": c.crop, "fr": c.fr, "en": c.en, "kind": c.kind} for c in CLASSES]


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="Photo d'une feuille (JPEG, PNG…)"),
    crop: Literal["cashew", "cassava", "maize", "tomato"] | None = Query(None, description="Culture, si connue"),
):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image trop lourde (10 Mo maximum).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, "Fichier illisible : envoie une image JPEG ou PNG.") from exc

    preds = predictor.predict(img, crop=crop, k=3)
    return {
        "crop_given": crop,
        "low_confidence": preds[0].prob < LOW_CONFIDENCE,
        "predictions": [
            {
                "id": p.class_id,
                "crop": CROPS[p.info.crop]["fr"],
                "label_fr": p.info.fr,
                "label_en": p.info.en,
                "kind": p.info.kind,
                "probability": round(p.prob, 4),
            }
            for p in preds
        ],
        "advice": list(preds[0].info.advice),
        "warning": GENERAL_WARNING,
    }
