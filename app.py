"""Démo web (Gradio) : point d'entrée pour Hugging Face Spaces.

Lancer en local :  python app.py   puis ouvrir http://localhost:7860

On envoie une photo de feuille ; le modèle renvoie les 3 diagnostics les plus probables,
avec des probabilités calibrées, et quelques conseils simples.
"""

from pathlib import Path

import gradio as gr

from leaf_core.labels import BY_ID, CROPS, GENERAL_WARNING
from leaf_core.model import LOW_CONFIDENCE, Predictor

ROOT = Path(__file__).parent
predictor = Predictor(ROOT / "model")

AUTO = "Je ne sais pas (détection automatique)"
CROP_CHOICES = [AUTO] + [v["fr"] for v in CROPS.values()]
CROP_BY_LABEL = {v["fr"]: k for k, v in CROPS.items()}


def label_of(class_id: str) -> str:
    c = BY_ID[class_id]
    return f"{CROPS[c.crop]['fr']} : {c.fr}"


def analyse(image, crop_label: str):
    if image is None:
        raise gr.Error("Ajoute d'abord une photo de feuille.")
    crop = CROP_BY_LABEL.get(crop_label)
    preds = predictor.predict(image, crop=crop, k=3)
    top = preds[0]
    info = top.info

    lines = [f"### Diagnostic probable : {label_of(top.class_id)} ({top.prob:.0%})"]
    if top.prob < LOW_CONFIDENCE:
        lines.append(
            "> ⚠️ **Le modèle hésite** (moins de 50 %). Reprends une photo nette d'une seule feuille, "
            "de près, à la lumière du jour, ou indique la culture. Regarde aussi les deux autres propositions."
        )
    if info.kind == "ravageur":
        lines.append("_Il s'agit de dégâts d'un ravageur (insecte ou acarien), pas d'une maladie._")
    lines.append("**Conseils :**")
    lines += [f"- {a}" for a in info.advice]
    lines.append(f"\n<small>{GENERAL_WARNING}</small>")
    return {label_of(p.class_id): p.prob for p in preds}, "\n".join(lines)


EXAMPLES = sorted(str(p) for p in (ROOT / "samples").glob("*.jpg"))

with gr.Blocks(title="Maladies des feuilles") as demo:
    gr.Markdown(
        "# Maladies des feuilles : maïs, manioc, tomate, anacarde\n"
        "Envoie la photo d'une feuille : le modèle propose les **3 diagnostics les plus probables** "
        "parmi 22 maladies, ravageurs ou état sain, avec quelques conseils. "
        "Les photos ne sont pas conservées.\n\n"
        "> **Prototype de recherche.** Sur des photos du Ghana proches de l'entraînement, le modèle a raison "
        "93 fois sur 100. Sur des photos d'autres pays et d'autres photographes, il a raison "
        "seulement **une à deux fois sur trois** (35 à 62 %) : c'est une piste, pas un diagnostic."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Photo de la feuille", type="pil", sources=["upload", "webcam"], height=360)
            crop = gr.Radio(CROP_CHOICES, value=AUTO, label="Culture (facultatif, améliore le résultat)")
            button = gr.Button("Analyser", variant="primary")
        with gr.Column():
            top3 = gr.Label(label="Top 3", num_top_classes=3)
            advice = gr.Markdown()
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=image, outputs=[top3, advice], label="Exemples (photos de test, jamais vues à l'entraînement)",
                    fn=lambda img: analyse(img, AUTO), run_on_click=True, cache_examples=False)
    gr.Markdown(
        "---\nProjet de **Haboudi BOUKARI**. Données d'entraînement : jeu **CCMT** (photos prises au champ "
        "au Ghana, Mensah et al., 2023, [CC BY 4.0](https://data.mendeley.com/datasets/bwh3zbpkpv/1)). "
        f"Modèle : `{predictor.meta['arch']}` pré-entraîné sur ImageNet ([timm](https://github.com/huggingface/pytorch-image-models)), "
        "affiné par mes soins. Les conseils sont des repères généraux, pas une prescription."
    )

    button.click(analyse, inputs=[image, crop], outputs=[top3, advice], concurrency_limit=2)
    image.upload(analyse, inputs=[image, crop], outputs=[top3, advice], concurrency_limit=2)


if __name__ == "__main__":
    demo.queue(max_size=20).launch(theme=gr.themes.Soft())
