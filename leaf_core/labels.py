"""Les 22 classes du modèle (jeu de données CCMT), leurs noms et des conseils simples.

Chaque classe a un identifiant « culture/etat » (ex. « maize/streak_virus »).
`folder` est le nom exact du dossier dans le jeu de données CCMT (fautes d'origine comprises).
`kind` vaut « maladie », « ravageur » ou « sain ».

Les conseils sont des repères généraux, pas une prescription : ils renvoient toujours
vers un conseiller agricole pour confirmer le diagnostic avant tout traitement.
"""

from dataclasses import dataclass, field

CROPS = {
    "cashew": {"fr": "Anacarde (cajou)", "en": "Cashew"},
    "cassava": {"fr": "Manioc", "en": "Cassava"},
    "maize": {"fr": "Maïs", "en": "Maize"},
    "tomato": {"fr": "Tomate", "en": "Tomato"},
}


@dataclass(frozen=True)
class LeafClass:
    id: str
    crop: str
    folder: str
    fr: str
    en: str
    kind: str
    advice: tuple = field(default_factory=tuple)


_HEALTHY_ADVICE = (
    "Aucun symptôme reconnu sur cette photo.",
    "Continuez à surveiller le champ chaque semaine, surtout les jeunes feuilles.",
)

CLASSES: list[LeafClass] = [
    # --- Anacarde ---
    LeafClass("cashew/anthracnose", "cashew", "anthracnose", "Anthracnose", "Anthracnose", "maladie", (
        "Champignon favorisé par l'humidité : taches brunes à noires sur jeunes feuilles et fleurs.",
        "Taillez et brûlez les rameaux atteints, ramassez les débris sous l'arbre, aérez la couronne.",
        "Un fongicide à base de cuivre peut être utile au moment des nouvelles pousses : demandez conseil.",
    )),
    LeafClass("cashew/gumosis", "cashew", "gumosis", "Gommose", "Gummosis", "maladie", (
        "Écoulement de gomme sur le tronc ou les branches, souvent après une blessure.",
        "Grattez l'écorce malade, couvrez la plaie (bouillie bordelaise ou pâte protectrice).",
        "Évitez de blesser les arbres (outils, feu, bétail).",
    )),
    LeafClass("cashew/healthy", "cashew", "healthy", "Sain", "Healthy", "sain", _HEALTHY_ADVICE),
    LeafClass("cashew/leaf_miner", "cashew", "leaf miner", "Mineuse des feuilles", "Leaf miner", "ravageur", (
        "Petite larve qui creuse des galeries dans les jeunes feuilles.",
        "Surveillez les nouvelles pousses ; taillez les rameaux très attaqués.",
        "Les produits à base de neem sont une option douce ; demandez conseil avant un insecticide.",
    )),
    LeafClass("cashew/red_rust", "cashew", "red rust", "Rouille rouge", "Red rust", "maladie", (
        "Taches rouille causées par une algue (Cephaleuros), en général peu grave.",
        "Taillez pour laisser passer la lumière et l'air.",
        "En cas de forte attaque, un produit à base de cuivre peut être conseillé.",
    )),
    # --- Manioc ---
    LeafClass("cassava/bacterial_blight", "cassava", "bacterial blight", "Bactériose", "Bacterial blight", "maladie", (
        "Bactérie : taches anguleuses d'aspect huileux, flétrissement, dessèchement des tiges.",
        "Plantez des boutures saines prises dans un champ sans maladie.",
        "Arrachez et brûlez les plants atteints, nettoyez les outils, faites une rotation des cultures.",
    )),
    LeafClass("cassava/brown_spot", "cassava", "brown spot", "Taches brunes (cercosporiose)", "Brown leaf spot", "maladie", (
        "Champignon : petites taches brunes sur les vieilles feuilles, souvent peu grave.",
        "Ramassez les feuilles tombées, espacez les plants, entretenez la fertilité du sol.",
        "Choisissez des variétés tolérantes.",
    )),
    LeafClass("cassava/green_mite", "cassava", "green mite", "Acarien vert", "Green mite", "ravageur", (
        "Petit acarien de saison sèche : jeunes feuilles tachetées de jaune, déformées et réduites.",
        "Préférez les variétés tolérantes ; l'acarien prédateur Typhlodromalus aripo aide à le contrôler.",
        "Évitez les insecticides à large spectre, qui tuent ses ennemis naturels.",
    )),
    LeafClass("cassava/healthy", "cassava", "healthy", "Sain", "Healthy", "sain", _HEALTHY_ADVICE),
    LeafClass("cassava/mosaic", "cassava", "mosaic", "Mosaïque du manioc", "Cassava mosaic disease", "maladie", (
        "Virus transmis par les mouches blanches et par les boutures : feuilles marbrées jaune-vert, déformées.",
        "Ne prenez jamais de boutures sur un plant malade ; utilisez des variétés résistantes.",
        "Arrachez et détruisez tôt les plants atteints.",
    )),
    # --- Maïs ---
    LeafClass("maize/fall_armyworm", "maize", "fall armyworm", "Chenille légionnaire d'automne", "Fall armyworm", "ravageur", (
        "Chenille qui mange le cornet : trous, feuilles déchiquetées, sciure humide dans le cornet.",
        "Inspectez le cornet des jeunes plants ; écrasez les œufs et les chenilles dans les petites parcelles.",
        "Produits biologiques (Bt, neem) possibles ; pour un insecticide, suivez l'étiquette et demandez conseil.",
    )),
    LeafClass("maize/grasshopper", "maize", "grasshoper", "Criquet / sauteriau", "Grasshopper", "ravageur", (
        "Dégâts de criquets : feuilles rongées depuis le bord.",
        "Surveillez les bordures du champ et désherbez-les (lieux de ponte).",
        "En cas d'invasion, alertez les services agricoles.",
    )),
    LeafClass("maize/healthy", "maize", "healthy", "Sain", "Healthy", "sain", _HEALTHY_ADVICE),
    LeafClass("maize/leaf_beetle", "maize", "leaf beetle", "Chrysomèle", "Leaf beetle", "ravageur", (
        "Petit coléoptère qui ronge la surface des feuilles (bandes claires).",
        "Désherbez autour du champ ; traitez seulement si les dégâts sont importants.",
        "Les extraits de neem sont une option douce.",
    )),
    LeafClass("maize/leaf_blight", "maize", "leaf blight", "Helminthosporiose (brûlure des feuilles)", "Leaf blight", "maladie", (
        "Champignon : longues taches grises ou brunes en forme de cigare.",
        "Utilisez des variétés résistantes et faites une rotation avec une légumineuse.",
        "Enfouissez ou retirez les résidus de récolte, évitez les semis trop serrés.",
    )),
    LeafClass("maize/leaf_spot", "maize", "leaf spot", "Taches foliaires", "Leaf spot", "maladie", (
        "Champignon : petites taches rectangulaires ou rondes sur les feuilles.",
        "Rotation des cultures, gestion des résidus, fumure équilibrée.",
        "Choisissez des variétés tolérantes.",
    )),
    LeafClass("maize/streak_virus", "maize", "streak virus", "Virus de la striure du maïs", "Maize streak virus", "maladie", (
        "Virus transmis par de petites cicadelles : fines stries jaunes le long des nervures.",
        "Arrachez et détruisez les jeunes plants atteints ; semez des variétés résistantes.",
        "Évitez les semis tardifs à côté d'un champ malade ; désherbez (les graminées hébergent les cicadelles).",
    )),
    # --- Tomate ---
    LeafClass("tomato/healthy", "tomato", "healthy", "Sain", "Healthy", "sain", _HEALTHY_ADVICE),
    LeafClass("tomato/leaf_blight", "tomato", "leaf blight", "Brûlure des feuilles", "Leaf blight", "maladie", (
        "Champignon ou oomycète (alternariose, mildiou) : taches brunes qui s'étendent, feuilles qui sèchent.",
        "Retirez les feuilles basses atteintes, arrosez au pied le matin, paillez le sol.",
        "Rotation : pas de tomate, piment, aubergine ou pomme de terre au même endroit pendant 2 à 3 ans.",
    )),
    LeafClass("tomato/leaf_curl", "tomato", "leaf curl", "Enroulement des feuilles (virus)", "Leaf curl", "maladie", (
        "Souvent un virus (TYLCV) transmis par les mouches blanches : feuilles enroulées, jaunies, plant nain.",
        "Arrachez tôt les plants atteints ; protégez la pépinière avec un filet anti-insectes.",
        "Pièges jaunes collants contre les mouches blanches, variétés résistantes, désherbage.",
    )),
    LeafClass("tomato/septoria_leaf_spot", "tomato", "septoria leaf spot", "Septoriose", "Septoria leaf spot", "maladie", (
        "Champignon : nombreuses petites taches rondes à centre clair sur les feuilles du bas.",
        "Retirez les feuilles atteintes, ne mouillez pas le feuillage, tuteurez et paillez.",
        "Faites une rotation des cultures.",
    )),
    LeafClass("tomato/verticillium_wilt", "tomato", "verticulium wilt", "Verticilliose (flétrissement)", "Verticillium wilt", "maladie", (
        "Champignon du sol : jaunissement en V des feuilles du bas, flétrissement. Pas de traitement curatif.",
        "Arrachez les plants atteints ; longue rotation avec des céréales ; variétés résistantes (marquées « V »).",
        "D'autres flétrissements (fusariose, bactérie) se ressemblent : faites confirmer le diagnostic.",
    )),
]

CLASS_IDS = [c.id for c in CLASSES]
BY_ID = {c.id: c for c in CLASSES}
BY_FOLDER = {(c.crop, c.folder): c for c in CLASSES}

CROP_FOLDERS = {"cashew": "Cashew", "cassava": "Cassava", "maize": "Maize", "tomato": "Tomato"}

GENERAL_WARNING = (
    "Ce modèle peut se tromper, surtout sur des photos très différentes de celles de son entraînement. "
    "Faites confirmer le diagnostic par un conseiller agricole (ATDA, INRAB, ONG) avant tout traitement, "
    "et respectez toujours l'étiquette des produits."
)


def crop_indices(crop: str) -> list[int]:
    """Indices des classes d'une culture, dans l'ordre de CLASS_IDS."""
    return [i for i, c in enumerate(CLASSES) if c.crop == crop]
