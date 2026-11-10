"""Correspondance entre les classes des jeux de test externes et nos 22 classes.

On ne garde que les classes qui ont un équivalent clair. « exact » : même maladie.
« approx » : même famille de symptômes, mais le nom CCMT est plus large ou ambigu
(ex. « leaf blight » de la tomate CCMT ne précise pas alternariose ou mildiou).
Les résultats sont donnés séparément pour les correspondances exactes.
"""

# PlantDoc (Singh et al., 2020, CC BY 4.0) : photos trouvées sur Internet, conditions réelles.
PLANTDOC = {
    "Corn leaf blight": ("maize/leaf_blight", "exact"),
    "Corn Gray leaf spot": ("maize/leaf_spot", "approx"),
    "Tomato Septoria leaf spot": ("tomato/septoria_leaf_spot", "exact"),
    "Tomato leaf": ("tomato/healthy", "exact"),
    "Tomato leaf yellow virus": ("tomato/leaf_curl", "exact"),
    "Tomato Early blight leaf": ("tomato/leaf_blight", "approx"),
    "Tomato leaf late blight": ("tomato/leaf_blight", "approx"),
}
# Classes PlantDoc écartées faute d'équivalent : Corn rust leaf, Tomato leaf bacterial spot,
# Tomato leaf mosaic virus, Tomato mold leaf, Tomato two spotted spider mites leaf.

# iCassava 2019 (Makerere AI Lab / NaCRRI, Ouganda) : photos de paysans au champ.
ICASSAVA = {
    "cbb": ("cassava/bacterial_blight", "exact"),
    "cgm": ("cassava/green_mite", "exact"),
    "cmd": ("cassava/mosaic", "exact"),
    "healthy": ("cassava/healthy", "exact"),
}
# Écartée : cbsd (striure brune, « brown streak ») : maladie virale différente
# de la cercosporiose (« brown spot ») de CCMT.
