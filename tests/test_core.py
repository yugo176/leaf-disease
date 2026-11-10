"""Tests rapides (CPU, sans téléchargement) : python -m pytest -q"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from leaf_core.labels import BY_FOLDER, CLASS_IDS, CLASSES, CROPS, crop_indices
from leaf_core.metrics import full_report, restrict_to_crop

ROOT = Path(__file__).resolve().parent.parent


def test_classes_coherentes():
    assert len(CLASSES) == 22 and len(set(CLASS_IDS)) == 22
    assert {c.crop for c in CLASSES} == set(CROPS)
    assert all(c.kind in {"maladie", "ravageur", "sain"} for c in CLASSES)
    assert all(c.advice for c in CLASSES)
    assert sum(len(crop_indices(k)) for k in CROPS) == 22
    assert len(BY_FOLDER) == 22


def test_culture_connue_interdit_les_autres_cultures():
    logits = np.zeros((2, 22))
    logits[:, CLASS_IDS.index("tomato/healthy")] = 10  # le modèle préfère la tomate…
    out = restrict_to_crop(logits, ["maize", "maize"])  # …mais l'utilisateur dit « maïs »
    assert all(CLASS_IDS[i].startswith("maize/") for i in out.argmax(1))


def test_rapport_parfait():
    y = np.arange(22)
    rep = full_report(np.eye(22) * 20, y, crops=[c.split("/")[0] for c in CLASS_IDS])
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0 and rep["culture_connue"]["accuracy"] == 1.0


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Un modèle aux poids aléatoires : on teste la mécanique, pas la qualité."""
    import torch
    from safetensors.torch import save_file

    from leaf_core.model import build_model

    d = tmp_path_factory.mktemp("model")
    arch = "mobilenetv3_large_100"
    torch.manual_seed(0)
    save_file({k: v.contiguous() for k, v in build_model(arch).state_dict().items()}, d / "model.safetensors")
    (d / "meta.json").write_text(json.dumps({"arch": arch, "classes": CLASS_IDS, "temperature": 1.5}))
    return d


def test_prediction_top3(tiny_model_dir):
    from leaf_core.model import Predictor

    p = Predictor(tiny_model_dir)
    img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
    preds = p.predict(img, k=3)
    assert len(preds) == 3 and preds[0].prob >= preds[1].prob >= preds[2].prob
    assert 0 < sum(x.prob for x in preds) <= 1.0001
    cassava = p.predict(img, crop="cassava", k=5)
    assert all(x.class_id.startswith("cassava/") for x in cassava)
    assert abs(sum(x.prob for x in cassava) - 1) < 1e-4  # 5 classes de manioc = 100 %


@pytest.mark.skipif(not (ROOT / "model" / "model.safetensors").exists(), reason="modèle entraîné absent")
def test_api_predict():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (40, 140, 40)).save(buf, "JPEG")
    r = client.post("/predict?crop=maize", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3 and all(p["id"].startswith("maize/") for p in body["predictions"])
    assert client.post("/predict", files={"file": ("x.txt", b"pas une image", "text/plain")}).status_code == 422
