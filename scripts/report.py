"""Transforme les résultats téléchargés de Modal en graphiques et en tableau Markdown.

    modal volume get leaf-disease-data runs results_modal/
    modal volume get leaf-disease-data results results_modal/
    python scripts/report.py            # écrit results/ et copie le meilleur modèle dans model/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from leaf_core.labels import BY_ID, CLASS_IDS, CROPS  # noqa: E402

SRC = ROOT / "results_modal"
OUT = ROOT / "results"
NICE = {"mobilenetv3_large_100": "MobileNetV3-Large", "efficientnet_b0": "EfficientNet-B0"}


def short(cid: str) -> str:
    c = BY_ID[cid]
    return f"{CROPS[c.crop]['fr'].split(' ')[0]} · {c.fr.split(' (')[0]}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def confusion_png(rep: dict, title: str, path: Path, rows_subset: list[int] | None = None):
    m = np.array(rep["confusion"]["matrix"], dtype=float)
    present = [i for i in range(len(CLASS_IDS)) if m[i].sum() > 0] if rows_subset is None else rows_subset
    cols = sorted(set(present) | {j for j in range(len(CLASS_IDS)) if m[present, j].sum() > 0})
    sub = m[np.ix_(present, cols)]
    norm = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.42 * len(cols), 1.2 + 0.42 * len(present)))
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [short(CLASS_IDS[j]) for j in cols], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(present)), [short(CLASS_IDS[i]) for i in present], fontsize=8)
    for i in range(len(present)):
        for j in range(len(cols)):
            if sub[i, j]:
                ax.text(j, i, int(sub[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Vraie classe")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    runs = sorted(p for p in (SRC / "runs").iterdir() if (p / "eval.json").exists())
    summary, lines = {}, []
    for rd in runs:
        ev = json.loads((rd / "eval.json").read_text(encoding="utf-8"))
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        hist = json.loads((rd / "history.json").read_text(encoding="utf-8"))
        summary[rd.name] = {"eval": ev, "meta": meta, "history": hist}

    # --- tableau comparatif
    lines += ["| Jeu de test | Mesure | " + " | ".join(NICE.get(k, k) for k in summary) + " |",
              "|---|---|" + "---|" * len(summary)]

    def row(label, measure, getter):
        vals = []
        for s in summary.values():
            try:
                vals.append(pct(getter(s["eval"])))
            except (KeyError, TypeError):
                vals.append("n/a")
        lines.append(f"| {label} | {measure} | " + " | ".join(vals) + " |")

    row("CCMT test (Ghana, mêmes champs)", "Précision (22 classes)", lambda e: e["ccmt_test"]["accuracy"])
    row("", "F1 macro", lambda e: e["ccmt_test"]["macro_f1"])
    row("", "Top 3", lambda e: e["ccmt_test"]["top3_accuracy"])
    row("", "Précision si culture indiquée", lambda e: e["ccmt_test"]["culture_connue"]["accuracy"])
    for name, label in (("plantdoc", "PlantDoc (Internet, maïs + tomate)"), ("icassava", "iCassava (Ouganda, manioc)")):
        row(label, "Précision (22 classes)", lambda e, n=name: e[n]["toutes_correspondances"]["accuracy"])
        row("", "Précision si culture indiquée", lambda e, n=name: e[n]["toutes_correspondances"]["culture_connue"]["accuracy"])
        row("", "Top 3", lambda e, n=name: e[n]["toutes_correspondances"]["top3_accuracy"])
        row("", "Correspondances exactes seulement, culture indiquée",
            lambda e, n=name: e[n]["correspondances_exactes"]["culture_connue"]["accuracy"])
    (OUT / "comparaison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # --- meilleur modèle : choisi sur la VALIDATION (jamais sur le test)
    best = max(summary, key=lambda k: summary[k]["meta"]["best_val_macro_f1"])
    print("\nModèle retenu (F1 macro de validation) :", best)
    ev = summary[best]["eval"]
    confusion_png(ev["ccmt_test"], f"{NICE.get(best, best)} : test CCMT (lignes normalisées)", OUT / "confusion_ccmt_test.png")
    confusion_png(ev["plantdoc"]["toutes_correspondances"], "Test externe PlantDoc", OUT / "confusion_plantdoc.png")
    confusion_png(ev["icassava"]["toutes_correspondances"], "Test externe iCassava", OUT / "confusion_icassava.png")

    # courbes d'apprentissage
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, s in summary.items():
        h = s["history"]
        ax.plot([x["epoch"] for x in h], [x["val_macro_f1"] for x in h], marker="o", ms=3, label=NICE.get(k, k))
    ax.set_xlabel("Époque")
    ax.set_ylabel("F1 macro (validation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "courbes_validation.png", dpi=130)
    plt.close(fig)

    # rapport par classe (test CCMT) pour le modèle retenu
    pc = ev["ccmt_test"]["par_classe"]
    tab = ["| Classe | Précision | Rappel | F1 | Images |", "|---|---|---|---|---|"]
    for cid in CLASS_IDS:
        if cid in pc:
            r = pc[cid]
            tab.append(f"| {short(cid)} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} | {int(r['support'])} |")
    (OUT / "par_classe_ccmt_test.md").write_text("\n".join(tab) + "\n", encoding="utf-8")

    # tests externes, classe par classe
    ext = ["| Jeu | Classe d'origine | Notre classe | Correspondance | Images | Rappel (22 classes) | Prédictions les plus fréquentes |",
           "|---|---|---|---|---|---|---|"]
    from leaf_core.external import ICASSAVA, PLANTDOC

    for name, mapping in (("plantdoc", PLANTDOC), ("icassava", ICASSAVA)):
        for sc, d in ev[name]["par_classe_source"].items():
            freq = ", ".join(f"{short(k)} ({v})" for k, v in d["predictions_les_plus_frequentes"].items())
            ext.append(f"| {name} | {sc} | {short(d['attendu'])} | {mapping[sc][1]} | {d['n']} | "
                       f"{pct(d['rappel_22_classes'])} | {freq} |")
    (OUT / "tests_externes_par_classe.md").write_text("\n".join(ext) + "\n", encoding="utf-8")

    json.dump({k: {"meta": v["meta"], "eval": {kk: vv for kk, vv in v["eval"].items()}} for k, v in summary.items()},
              open(OUT / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if (SRC / "results" / "split_report.json").exists():
        shutil.copy(SRC / "results" / "split_report.json", OUT / "split_report.json")

    # copie du modèle retenu pour la démo et l'API
    (ROOT / "model").mkdir(exist_ok=True)
    for f in ("model.safetensors", "meta.json"):
        shutil.copy(SRC / "runs" / best / f, ROOT / "model" / f)
    print("modèle copié dans model/")


if __name__ == "__main__":
    main()
