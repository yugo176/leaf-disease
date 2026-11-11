"""Lance une étape du pipeline sur l'app Modal DÉPLOYÉE, puis attend sa fin.

Pourquoi pas simplement `modal run` ? Avec `modal run`, si la connexion du PC coupe
(c'est arrivé pendant ce projet), Modal annule le travail en cours. Une app déployée
continue de tourner quoi qu'il arrive au PC ; ce script peut même être relancé
avec `--wait <id>` pour reprendre l'attente.

    modal deploy modal_pipeline.py           # (avec LEAF_GPU=none pour tout faire sur CPU)
    python scripts/spawn.py fetch
    python scripts/spawn.py split
    python scripts/spawn.py train            # les deux modèles en parallèle
    python scripts/spawn.py evaluate
    python scripts/spawn.py --wait fc-XXXX   # reprendre l'attente d'un appel déjà lancé
"""

import sys
import time

import modal

APP = "leaf-disease"
ARCHS = ["mobilenetv3_large_100.ra_in1k", "efficientnet_b0.ra_in1k"]
STEPS = {
    "fetch": [("fetch_all", ())],
    "split": [("make_split", ())],
    "train": [("train_one", (a, 15)) for a in ARCHS],
    "evaluate": [("evaluate_all", ())],
}


def wait(call_id: str):
    while True:
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=60)
        except TimeoutError:
            continue
        except Exception as e:  # coupure réseau du PC : on réessaie, le travail continue chez Modal
            if "Connection" in type(e).__name__:
                print("  (réseau indisponible, nouvel essai dans 30 s)")
                time.sleep(30)
                continue
            raise


def main(argv):
    if argv[:1] == ["--wait"]:
        ids = argv[1:]
    else:
        ids = []
        for fn, args in STEPS[argv[0]]:
            call = modal.Function.from_name(APP, fn).spawn(*args)
            print(f"{fn}{args} lancé : {call.object_id}")
            ids.append(call.object_id)
    for i in ids:
        print(i, "→", wait(i))


if __name__ == "__main__":
    main(sys.argv[1:])
