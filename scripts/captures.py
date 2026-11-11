"""Captures d'écran de la démo pour le portfolio (démo lancée sur http://localhost:7860).

    pip install playwright        # utilise le navigateur Edge déjà installé, rien d'autre à télécharger
    python app.py                 # dans un autre terminal
    python scripts/captures.py    # écrit docs/captures/*.png
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "captures"
URL = "http://localhost:7860"

# (nom du fichier, photo d'exemple, culture à cocher ou None)
SHOTS = [
    ("1_mais_striure.png", "maize__streak_virus.jpg", None),
    ("2_manioc_mosaique.png", "cassava__mosaic.jpg", "Manioc"),
    ("3_tomate_erreur_septoriose.png", "tomato__septoria_leaf_spot.jpg", None),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=2)
        page.goto(URL)
        page.wait_for_selector("text=Analyser")
        page.screenshot(path=OUT / "0_accueil.png", full_page=True)
        for name, sample, crop in SHOTS:
            page.goto(URL)
            page.wait_for_selector("text=Analyser")
            if crop:
                page.get_by_label(crop, exact=True).check()
            page.set_input_files("input[type=file]", ROOT / "samples" / sample)
            page.wait_for_selector("text=Diagnostic probable", timeout=30000)
            page.wait_for_timeout(800)
            page.screenshot(path=OUT / name, full_page=True)
            print("écrit", OUT / name)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "4_mobile.png", full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
