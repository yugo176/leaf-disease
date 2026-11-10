"""Métriques d'évaluation, calculées à partir des logits (sorties brutes du réseau)."""

from __future__ import annotations

import numpy as np

from .labels import CLASS_IDS, crop_indices


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Écart moyen entre la confiance annoncée et la précision réelle (0 = parfait)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def restrict_to_crop(logits: np.ndarray, crops: list[str]) -> np.ndarray:
    """Mode « culture connue » : on interdit les classes des autres cultures."""
    out = np.full_like(logits, -np.inf)
    for crop in set(crops):
        rows = np.array([c == crop for c in crops])
        idx = crop_indices(crop)
        out[np.ix_(rows, idx)] = logits[np.ix_(rows, idx)]
    return out


def full_report(logits: np.ndarray, y: np.ndarray, crops: list[str] | None = None) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    y = np.asarray(y)
    if len(y) == 0:
        return {"n": 0}
    probs = _softmax(logits)
    pred = probs.argmax(1)
    present = sorted(set(y.tolist()))
    top3 = np.argsort(-probs, 1)[:, :3]
    rep = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "top3_accuracy": float((top3 == y[:, None]).any(1).mean()),
        "ece": expected_calibration_error(probs, y),
        "bonne_culture": float(np.mean([CLASS_IDS[p].split("/")[0] == CLASS_IDS[t].split("/")[0]
                                        for p, t in zip(pred, y)])),
        "par_classe": {
            CLASS_IDS[int(k)]: v for k, v in classification_report(
                y, pred, labels=present, output_dict=True, zero_division=0).items() if k.isdigit()
        },
        "confusion": {
            "labels": [CLASS_IDS[i] for i in range(len(CLASS_IDS))],
            "matrix": confusion_matrix(y, pred, labels=list(range(len(CLASS_IDS)))).tolist(),
        },
    }
    if crops is not None:
        lc = restrict_to_crop(logits, crops)
        pc = lc.argmax(1)
        rep["culture_connue"] = {
            "accuracy": float((pc == y).mean()),
            "macro_f1": float(f1_score(y, pc, labels=present, average="macro", zero_division=0)),
        }
    return rep
