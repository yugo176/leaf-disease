---
title: Maladies des feuilles
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Diagnostic de maladies du maïs, manioc, tomate, anacarde
---

# Maladies des feuilles : maïs, manioc, tomate, anacarde

Un modèle de vision qui reconnaît, à partir de la photo d'une feuille, **22 maladies, ravageurs
ou états sains** sur quatre cultures importantes au Bénin et en Afrique de l'Ouest :
le **maïs**, le **manioc**, la **tomate** et l'**anacarde** (cajou). Il est livré avec une démo web
(Gradio) et une petite API (FastAPI), et tourne sur un simple processeur.

- **Auteur :** Haboudi BOUKARI, élève-ingénieur (informatique, logistique et recherche opérationnelle), ENSGMM
- **Statut :** prototype de recherche, entraîné et évalué. Démo locale ; pas encore publiée en ligne.

<!-- RESULTATS -->

## Pourquoi ce projet

Au Bénin, le maïs, le manioc, la tomate et l'anacarde font vivre beaucoup de petits producteurs.
Une maladie repérée tôt (mosaïque du manioc, striure du maïs, chenille légionnaire…) se gère mieux,
mais un conseiller agricole n'est pas toujours disponible. L'idée : une photo prise au téléphone
donne une **première piste**, à confirmer ensuite par un conseiller.

Le piège classique de ce genre de projet est de s'entraîner sur **PlantVillage** (photos de
laboratoire, une feuille sur fond uni) : on obtient plus de 99 % de précision en test,
puis le modèle s'effondre sur une vraie photo prise au champ. J'ai donc choisi
des **photos prises au champ en Afrique de l'Ouest**, et je mesure le modèle sur des photos
**d'autres pays et d'autres photographes**.

## Données et licences

| Jeu de données | Rôle | Contenu | Licence |
|---|---|---|---|
| [CCMT](https://data.mendeley.com/datasets/bwh3zbpkpv/1) (Mensah et al., 2023, *Data in Brief*) | entraînement, validation, test | 24 881 photos brutes prises dans des fermes du Ghana, validées par des phytopathologistes, 22 classes | CC BY 4.0 |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh et al., 2020) | **test externe** maïs et tomate | photos trouvées sur Internet, conditions réelles | CC BY 4.0 |
| [iCassava 2019](https://www.tensorflow.org/datasets/catalog/cassava) (Makerere AI Lab, NaCRRI) | **test externe** manioc | photos de paysans ougandais au champ | non précisée : utilisé **seulement pour tester**, jamais redistribué |

Les images ne sont **pas** dans ce dépôt : elles sont téléchargées depuis leur source par le pipeline.
Les quelques photos du dossier `samples/` viennent du jeu de test CCMT (CC BY 4.0, Mensah et al., 2023),
réduites à 512 px.

### Les 22 classes

| Culture | Maladies | Ravageurs | |
|---|---|---|---|
| Anacarde | anthracnose, gommose, rouille rouge | mineuse des feuilles | sain |
| Manioc | bactériose, taches brunes, mosaïque | acarien vert | sain |
| Maïs | helminthosporiose, taches foliaires, virus de la striure | chenille légionnaire, criquet, chrysomèle | sain |
| Tomate | brûlure des feuilles, enroulement (virus), septoriose, verticilliose | | sain |

## Méthode

<!-- METHODE -->

## Lancer le projet

### Démo et API (CPU, en local)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (sous Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
python app.py                     # démo : http://localhost:7860
uvicorn api.main:app --port 8000  # API : http://localhost:8000/docs
```

Exemple d'appel de l'API :

```bash
curl -F "file=@samples/cassava__mosaic.jpg" "http://localhost:8000/predict?crop=cassava"
```

### Refaire l'entraînement (Modal)

```bash
pip install -r requirements-dev.txt
modal setup                                          # connexion au compte Modal (une fois)
set LEAF_GPU=none                                    # PowerShell : $env:LEAF_GPU="none" ; ou LEAF_GPU=L4 avec un GPU
modal run modal_pipeline.py::fetch                   # télécharge les données dans un Volume Modal
modal run modal_pipeline.py::split                   # doublons + découpage
modal run modal_pipeline.py::train                   # entraîne les deux modèles en parallèle
modal run modal_pipeline.py::evaluate                # métriques et tests externes
modal volume get leaf-disease-data runs results_modal/
modal volume get leaf-disease-data results results_modal/
python scripts/report.py                             # graphiques, tableaux, copie du modèle retenu
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Structure

```
leaf_core/          cœur partagé : classes et conseils, modèle et prédiction, métriques
modal_pipeline.py   données → découpage → entraînement → évaluation, sur Modal
scripts/report.py   graphiques et tableaux à partir des résultats
app.py              démo Gradio (point d'entrée Hugging Face Spaces)
api/main.py         API FastAPI : POST /predict
model/              modèle retenu (poids safetensors + meta.json)
results/            métriques, matrices de confusion, courbes
samples/            photos d'exemple (jeu de test CCMT)
tests/              tests rapides (python -m pytest -q)
```

## Licence

Code sous licence Apache 2.0. Les poids du modèle sont dérivés de CCMT (CC BY 4.0) :
toute réutilisation doit citer Mensah et al. (2023) :

> Mensah, P. K. et al. (2023). *CCMT: Dataset for crop pest and disease detection*. Data in Brief, 49, 109306.
> https://doi.org/10.1016/j.dib.2023.109306
