"""Pipeline complet sur Modal : données → découpage → entraînement → évaluation.

Méthode robuste (le travail continue même si le PC perd sa connexion) :

    set LEAF_GPU=none                 # PowerShell : $env:LEAF_GPU="none" ; ou L4 si le compte a une carte
    modal deploy modal_pipeline.py
    python scripts/spawn.py fetch     # télécharge CCMT, PlantDoc, iCassava ; une archive par jeu
    python scripts/spawn.py split     # doublons + découpage entraînement / validation / test
    python scripts/spawn.py train     # fine-tuning des deux modèles (en parallèle)
    python scripts/spawn.py evaluate  # métriques, calibration, tests externes

Puis, pour récupérer les résultats :

    modal volume get leaf-disease-data runs results_modal/
    modal volume get leaf-disease-data results results_modal/
"""

from __future__ import annotations

import os

import modal

APP_NAME = "leaf-disease"
VOL_PATH = "/vol"
# GPU Modal : LEAF_GPU=T4 / L4 / A10… ; LEAF_GPU=none pour les étapes sans GPU (données, découpage)
_gpu_env = os.environ.get("LEAF_GPU", "L4")
GPU = None if _gpu_env.lower() in ("none", "cpu", "") else _gpu_env
# Sans GPU, on prend beaucoup de cœurs CPU : 8 pour lire les images, le reste pour le calcul.
CPU_ONLY_CORES = 32
TRAIN_CPU = 8 if GPU else CPU_ONLY_CORES
LOADER_WORKERS = 8

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("leaf-disease-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "timm==1.0.31",
        "safetensors==0.8.0",
        "pillow==12.3.0",
        "numpy==2.4.6",
        "scikit-learn==1.9.1",
        "imagehash==4.3.2",
        "requests",
        "matplotlib",
    )
    # Même réglage GPU/CPU à l'intérieur du conteneur qu'au moment du déploiement
    .env({"LEAF_GPU": os.environ.get("LEAF_GPU", "L4")})
    .add_local_python_source("leaf_core")
)

CCMT_ZIP = "https://data.mendeley.com/public-api/zip/bwh3zbpkpv/download/1"
PLANTDOC_ZIP = "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master"
ICASSAVA_ZIP = "https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip"

SHORT_SIDE = 320  # on réduit chaque photo une fois : plus rapide à lire pendant l'entraînement
ARCHS = ["mobilenetv3_large_100.ra_in1k", "efficientnet_b0.ra_in1k"]


# ----------------------------------------------------------------------------- outils


def _download(url: str, dest: str) -> None:
    import requests

    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "leaf-disease-research"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (512 << 20) < (8 << 20):
                    print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} Go")
    print(f"téléchargé : {dest} ({done / 1e9:.2f} Go)")


def _resize_and_hash(args):
    """Lit une image (octets), la réduit, l'enregistre et calcule ses empreintes."""
    import hashlib
    import io

    import imagehash
    from PIL import Image, ImageOps

    data, out_path = args
    sha = hashlib.sha256(data).hexdigest()
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # image illisible : on la signale et on l'écarte
        return {"out": out_path, "sha256": sha, "error": repr(e)}
    w, h = img.size
    s = min(w, h)
    if s > SHORT_SIDE:
        img = img.resize((round(w * SHORT_SIDE / s), round(h * SHORT_SIDE / s)), Image.BICUBIC)
    img.save(out_path, "JPEG", quality=92)
    return {"out": out_path, "sha256": sha, "w": w, "h": h, "phash": str(imagehash.phash(img))}


def _process_zip(zip_path: str, select, out_root: str, workers: int = 8) -> list[dict]:
    """Extrait du zip les fichiers choisis par `select(parts) -> (label, meta) | None`."""
    import os
    import zipfile
    from concurrent.futures import ProcessPoolExecutor

    rows, jobs = [], []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"{len(names)} images dans le zip")
        for n in names:
            parts = n.replace("\\", "/").split("/")
            sel = select(parts)
            if sel is None:
                continue
            label, meta = sel
            d = os.path.join(out_root, label.replace("/", "__"))
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f"{len(jobs):06d}.jpg")
            jobs.append((z.read(n), out))
            rows.append({"label": label, "src": n, **meta})
    with ProcessPoolExecutor(workers) as ex:
        for row, res in zip(rows, ex.map(_resize_and_hash, jobs, chunksize=64)):
            row.update(res)
    return rows


def _write_csv(rows: list[dict], path: str) -> None:
    import csv

    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: str) -> list[dict]:
    import csv

    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- 1. données


def _sel_ccmt(parts):
    from leaf_core.labels import BY_FOLDER, CROP_FOLDERS

    crop_by_folder = {v: k for k, v in CROP_FOLDERS.items()}
    if "Raw Data" not in parts or len(parts) < 4:  # on ignore la version « augmentée »
        return None
    crop, folder = parts[-3], parts[-2]
    if crop not in crop_by_folder:
        return None
    c = BY_FOLDER.get((crop_by_folder[crop], folder))
    if c is None:
        print("dossier inconnu :", parts)
        return None
    return c.id, {"crop": c.crop}


def _sel_plantdoc(parts):
    from leaf_core.external import PLANTDOC

    if len(parts) < 3 or parts[-3] not in ("train", "test") or parts[-2] not in PLANTDOC:
        return None
    label, match = PLANTDOC[parts[-2]]
    return label, {"src_class": parts[-2], "match": match, "src_split": parts[-3]}


def _sel_icassava(parts):
    from leaf_core.external import ICASSAVA

    if len(parts) < 2 or parts[-2] not in ICASSAVA:
        return None
    label, match = ICASSAVA[parts[-2]]
    return label, {"src_class": parts[-2], "match": match, "src_split": parts[-3] if len(parts) >= 3 else ""}


DATASETS = {
    "ccmt": (CCMT_ZIP, _sel_ccmt),
    "plantdoc": (PLANTDOC_ZIP, _sel_plantdoc),
    "icassava": (ICASSAVA_ZIP, _sel_icassava),
}


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=8, memory=16384, timeout=3 * 3600)
def fetch_all():
    """Télécharge chaque jeu, réduit les images sur le disque LOCAL du conteneur,
    puis écrit sur le Volume un manifeste CSV et UNE archive tar.

    Pourquoi une archive : relire des milliers de petits fichiers sur un Volume Modal
    s'est révélé très lent (~5 fichiers/s), alors qu'un seul gros fichier se lit vite.
    """
    import collections
    import os
    import shutil
    import tarfile
    import time

    os.makedirs("/tmp/dl", exist_ok=True)
    for name, (url, select) in DATASETS.items():
        tar_path = f"{VOL_PATH}/{name}/img.tar"
        if os.path.exists(tar_path):
            continue
        zip_path = f"/tmp/dl/{name}.zip"
        _download(url, zip_path)
        local = f"/tmp/{name}/img"
        rows = _process_zip(zip_path, select, local)
        os.remove(zip_path)
        for r in rows:  # chemins « logiques » : _unpack les remplace par le disque local
            r["out"] = r["out"].replace(local, f"{VOL_PATH}/{name}/img", 1)
        man = f"{VOL_PATH}/{name}/manifest.csv"
        if os.path.exists(man):  # manifeste déjà utilisé par le découpage : on vérifie qu'il est identique
            old = {r["out"]: r["sha256"] for r in _read_csv(man)}
            new = {r["out"]: r["sha256"] for r in rows}
            assert old == new, f"{name} : les images reconstruites ne correspondent pas au manifeste"
            print(f"{name} : {len(new)} images identiques au manifeste existant")
        else:
            os.makedirs(os.path.dirname(man), exist_ok=True)
            _write_csv(rows, man)
        print(name, ":", collections.Counter(r.get("src_class", r["label"]) for r in rows))
        t = time.time()
        with tarfile.open("/tmp/dl/img.tar", "w") as tar:
            tar.add(local, arcname=".")
        shutil.copyfile("/tmp/dl/img.tar", tar_path)
        os.remove("/tmp/dl/img.tar")
        vol.commit()
        print(f"{name} : archive écrite en {time.time() - t:.0f} s ({os.path.getsize(tar_path) / 1e6:.0f} Mo)")


@app.local_entrypoint()
def fetch():
    fetch_all.remote()


# ----------------------------------------------------------------------------- 2. découpage


PHASH_MAX_DIST = 2  # ≤ 2 bits d'écart sur 64 : même photo. À 4, on relie déjà des cultures différentes (faux positifs)


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=16384, timeout=3600)
def make_split(seed: int = 42):
    """Regroupe doublons et quasi-doublons, puis découpe 70 / 15 / 15 par groupe.

    Un groupe (même photo, ou presque) va entièrement d'un seul côté : sinon le modèle
    serait testé sur des images qu'il a déjà vues, et le score serait gonflé.
    """
    import collections
    import json

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    from leaf_core.labels import CLASS_IDS

    rows = [r for r in _read_csv(f"{VOL_PATH}/ccmt/manifest.csv") if not r.get("error")]
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    # doublons exacts (mêmes octets)
    by_sha = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_sha[r["sha256"]].append(i)
    exact_dups = sum(len(v) - 1 for v in by_sha.values())
    for v in by_sha.values():
        for j in v[1:]:
            union(v[0], j)

    # quasi-doublons (empreinte perceptuelle proche)
    h = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    near_pairs = 0
    for start in range(0, n, 2000):
        block = h[start:start + 2000, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= PHASH_MAX_DIST)
        for i, j in zip(ii + start, jj):
            if i < j:
                near_pairs += 1
                union(int(i), int(j))

    groups = np.array([find(i) for i in range(n)])
    labels_per_group = collections.defaultdict(set)
    for g, r in zip(groups, rows):
        labels_per_group[g].add(r["label"])
    conflict = {g for g, s in labels_per_group.items() if len(s) > 1}
    keep = [i for i in range(n) if groups[i] not in conflict]
    dropped_conflict = n - len(keep)

    y = np.array([CLASS_IDS.index(rows[i]["label"]) for i in keep])
    g = groups[keep]
    # test ≈ 15 %
    sgk = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgk.split(np.zeros(len(y)), y, g))
    # validation ≈ 15 % du total, prise dans le reste
    sgk2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr, va = next(sgk2.split(np.zeros(len(rest_idx)), y[rest_idx], g[rest_idx]))
    split = {}
    for k in rest_idx[tr]:
        split[keep[k]] = "train"
    for k in rest_idx[va]:
        split[keep[k]] = "val"
    for k in test_idx:
        split[keep[k]] = "test"

    out = []
    for i, r in enumerate(rows):
        if i in split:
            out.append({"path": r["out"], "label": r["label"], "crop": r["crop"],
                        "group": int(groups[i]), "split": split[i]})
    _write_csv(out, f"{VOL_PATH}/split.csv")

    sizes = collections.Counter(r["split"] for r in out)
    per_class = {c: dict(collections.Counter(r["split"] for r in out if r["label"] == c)) for c in CLASS_IDS}
    report = {
        "images_lisibles": n,
        "images_illisibles": len(_read_csv(f"{VOL_PATH}/ccmt/manifest.csv")) - n,
        "doublons_exacts": exact_dups,
        "paires_quasi_doublons": near_pairs,
        "groupes": int(len(set(groups.tolist()))),
        "images_ecartees_etiquettes_contradictoires": dropped_conflict,
        "tailles": dict(sizes),
        "par_classe": per_class,
        "seuil_phash": PHASH_MAX_DIST,
    }
    import os
    os.makedirs(f"{VOL_PATH}/results", exist_ok=True)
    with open(f"{VOL_PATH}/results/split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    vol.commit()
    print(json.dumps({k: v for k, v in report.items() if k != "par_classe"}, ensure_ascii=False, indent=2))
    return report


@app.local_entrypoint()
def split():
    make_split.remote()


# ----------------------------------------------------------------------------- 3. entraînement


class _DS:
    def __init__(self, rows, tf, root_from="", root_to=""):
        from leaf_core.labels import CLASS_IDS

        self.paths = [r["path"].replace(root_from, root_to, 1) if root_from else r["path"] for r in rows]
        self.y = [CLASS_IDS.index(r["label"]) for r in rows]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


def _train_transform(size=224):
    import torch
    from torchvision.transforms import v2 as T

    from leaf_core.model import MEAN, STD

    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(25)], p=0.5),
        T.ColorJitter(0.3, 0.3, 0.3, 0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.15),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def _unpack(name: str):
    """Extrait l'archive d'un jeu sur le disque local du conteneur ; renvoie (ancien, nouveau) préfixe."""
    import tarfile
    import time

    t = time.time()
    with open(f"{VOL_PATH}/{name}/img.tar", "rb") as fh:  # lecture séquentielle d'un seul gros fichier
        with tarfile.open(fileobj=fh, mode="r|") as tar:
            tar.extractall(f"/tmp/{name}/img", filter="data")
    print(f"{name} : images extraites en {time.time() - t:.0f} s")
    return f"{VOL_PATH}/{name}/img", f"/tmp/{name}/img"


def _copy_images_locally():
    return _unpack("ccmt")


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    import os

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else TRAIN_CPU
    # sans GPU, on a réservé CPU_ONLY_CORES cœurs (ne pas se fier à TRAIN_CPU, calculé à l'import)
    torch.set_num_threads(max(1, min(cores, CPU_ONLY_CORES) - LOADER_WORKERS))
    return "cpu"


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=GPU, cpu=TRAIN_CPU, memory=32768, timeout=6 * 3600)
def train_one(arch: str, epochs: int = 20, lr: float = 1e-3, batch: int = 64, seed: int = 0):
    rows = _read_csv(f"{VOL_PATH}/split.csv")
    src, dst = _copy_images_locally()
    meta = train_core(rows, src, dst, f"{VOL_PATH}/runs/{arch.split('.')[0]}", arch, epochs, lr, batch, seed)
    vol.commit()
    return meta


def train_core(rows, src, dst, out_dir, arch, epochs=20, lr=1e-3, batch=64, seed=0, pretrained=True):
    """Boucle d'entraînement, indépendante de Modal (testable en local)."""
    import json
    import os
    import time

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from sklearn.metrics import f1_score

    from leaf_core.labels import CLASS_IDS
    from leaf_core.model import IMG_SIZE, RESIZE, build_model, eval_transform

    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    ds_tr = _DS(tr_rows, _train_transform(IMG_SIZE), src, dst)
    ds_va = _DS(va_rows, eval_transform(), src, dst)
    dev = _device()
    gpu = dev == "cuda"
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch, shuffle=True, num_workers=LOADER_WORKERS, pin_memory=gpu,
                                        drop_last=True, persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(ds_va, 128, num_workers=LOADER_WORKERS, pin_memory=gpu,
                                        persistent_workers=True)

    counts = np.bincount(ds_tr.y, minlength=len(CLASS_IDS))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.mean()  # classes rares un peu plus pesées (racine carrée : compromis doux)

    # Sur GPU : calcul en bfloat16 (plus rapide). Sur CPU : float32 classique.
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if gpu else __import__("contextlib").nullcontext()
    print(f"appareil : {dev}, threads : {torch.get_num_threads()}, images d'entraînement : {len(ds_tr)}")
    model = build_model(arch, pretrained=pretrained).to(dev).to(memory_format=torch.channels_last)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    os.makedirs(out_dir, exist_ok=True)
    history, best = [], -1.0
    for ep in range(epochs):
        t = time.time()
        model.train()
        tot, n = 0.0, 0
        for step, (x, y) in enumerate(dl_tr):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with amp:
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
            if ep == 0 and step in (10, 50):
                print(f"  étape {step} : {n / (time.time() - t):.0f} images/s")
        model.eval()
        preds, ys = [], []
        with torch.no_grad(), amp:
            for x, y in dl_va:
                preds.append(model(x.to(dev).to(memory_format=torch.channels_last)).argmax(1).cpu())
                ys.append(y)
        p, yv = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        acc, f1 = float((p == yv).mean()), float(f1_score(yv, p, average="macro"))
        history.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": acc, "val_macro_f1": f1,
                        "seconds": round(time.time() - t, 1)})
        print(history[-1])
        if f1 > best:
            best = f1
            sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(sd, f"{out_dir}/model.safetensors")
    meta = {"arch": arch, "classes": CLASS_IDS, "img_size": IMG_SIZE, "resize": RESIZE,
            "epochs": epochs, "lr": lr, "batch": batch, "seed": seed, "best_val_macro_f1": best,
            "device": GPU or f"CPU {TRAIN_CPU} cœurs", "train_images": len(ds_tr), "val_images": len(ds_va)}
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(f"{out_dir}/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return meta


@app.local_entrypoint()
def train(epochs: int = 20):
    for meta in train_one.starmap([(a, epochs) for a in ARCHS]):
        print(meta["arch"], "meilleur F1 macro (validation) :", round(meta["best_val_macro_f1"], 4))


# ----------------------------------------------------------------------------- 4. évaluation


def _logits(model, rows, root_from="", root_to="", feats=False):
    import torch

    from leaf_core.model import eval_transform

    ds = _DS(rows, eval_transform(), root_from, root_to)
    dev = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(ds, 128, num_workers=LOADER_WORKERS)
    L, F = [], []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev).to(memory_format=torch.channels_last)
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            L.append(model.get_classifier()(f).float().cpu())
            if feats:
                F.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    import numpy as np

    return np.asarray(torch.cat(L)), (np.asarray(torch.cat(F)) if feats else None), np.array(ds.y)


def _fit_temperature(logits, y):
    """Calibration : un seul nombre T tel que softmax(logits / T) reflète mieux la fiabilité."""
    import torch

    lg, yt = torch.tensor(logits), torch.tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=GPU, cpu=TRAIN_CPU, memory=32768, timeout=3 * 3600)
def evaluate_all():
    import json
    import os

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from leaf_core.labels import CLASS_IDS
    from leaf_core.metrics import full_report

    rows = _read_csv(f"{VOL_PATH}/split.csv")
    src, dst = _copy_images_locally()
    sets = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    ext = {name: _read_csv(f"{VOL_PATH}/{name}/manifest.csv") for name in ("plantdoc", "icassava")}
    for name in ext:
        a, b = _unpack(name)
        ext[name] = [{"path": r["out"].replace(a, b, 1), "label": r["label"], "match": r["match"],
                      "src_class": r["src_class"]} for r in ext[name] if not r.get("error")]

    summary = {}
    for arch_dir in sorted(os.listdir(f"{VOL_PATH}/runs")):
        rd = f"{VOL_PATH}/runs/{arch_dir}"
        meta = json.load(open(f"{rd}/meta.json"))
        model = __import__("leaf_core.model", fromlist=["build_model"]).build_model(meta["arch"])
        model.load_state_dict(load_file(f"{rd}/model.safetensors"))
        model = model.to(_device()).eval().to(memory_format=torch.channels_last)

        lv, _, yv = _logits(model, sets["val"], src, dst)
        T = _fit_temperature(lv, yv)
        meta["temperature"] = T
        json.dump(meta, open(f"{rd}/meta.json", "w"), indent=2)

        lt, ft, yt = _logits(model, sets["test"], src, dst, feats=True)
        _, ftr, _ = _logits(model, sets["train"], src, dst, feats=True)
        # vérification de fuite : chaque image de test a-t-elle une « jumelle » dans l'entraînement ?
        nn_sim = np.concatenate([(torch.tensor(ft[i:i + 512]) @ torch.tensor(ftr).T).max(1).values.numpy()
                                 for i in range(0, len(ft), 512)])

        res = {"temperature": T, "ccmt_val": full_report(lv / T, yv),
               "ccmt_test": full_report(lt / T, yt, crops=[r["crop"] for r in sets["test"]])}
        near = nn_sim >= 0.95
        res["ccmt_test"]["fuite"] = {
            "part_tests_avec_voisin_tres_proche": float(near.mean()),
            "precision_sur_ces_images": float(((lt.argmax(1) == yt)[near]).mean()) if near.any() else None,
            "precision_sur_les_autres": float(((lt.argmax(1) == yt)[~near]).mean()) if (~near).any() else None,
        }
        preds = {}
        for name, erows in ext.items():
            le, _, ye = _logits(model, erows)
            le = le / T
            crops = [CLASS_IDS[i].split("/")[0] for i in ye]
            exact = np.array([r["match"] == "exact" for r in erows])
            res[name] = {
                "toutes_correspondances": full_report(le, ye, crops=crops),
                "correspondances_exactes": full_report(le[exact], ye[exact], crops=[c for c, e in zip(crops, exact) if e]),
                "par_classe_source": {},
            }
            for sc in sorted({r["src_class"] for r in erows}):
                m = np.array([r["src_class"] == sc for r in erows])
                pred = le[m].argmax(1)
                res[name]["par_classe_source"][sc] = {
                    "n": int(m.sum()),
                    "attendu": CLASS_IDS[ye[m][0]],
                    "rappel_22_classes": float((pred == ye[m]).mean()),
                    "predictions_les_plus_frequentes": {
                        CLASS_IDS[k]: int(v) for k, v in
                        sorted(zip(*np.unique(pred, return_counts=True)), key=lambda t: -t[1])[:3]
                    },
                }
            preds[name] = {"logits": le, "y": ye}
        np.savez_compressed(f"{rd}/logits.npz", val=lv / T, val_y=yv, test=lt / T, test_y=yt,
                            **{f"{k}": v["logits"] for k, v in preds.items()},
                            **{f"{k}_y": v["y"] for k, v in preds.items()})
        json.dump(res, open(f"{rd}/eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        summary[arch_dir] = res
        print(arch_dir, json.dumps({
            "test_acc": res["ccmt_test"]["accuracy"], "test_f1": res["ccmt_test"]["macro_f1"],
            "plantdoc_acc": res["plantdoc"]["toutes_correspondances"]["accuracy"],
            "icassava_acc": res["icassava"]["toutes_correspondances"]["accuracy"],
        }))

    os.makedirs(f"{VOL_PATH}/results", exist_ok=True)
    json.dump(summary, open(f"{VOL_PATH}/results/eval_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    vol.commit()


@app.local_entrypoint()
def evaluate():
    evaluate_all.remote()


# ----------------------------------------------------------------------------- 5. exemples pour la démo

SAMPLE_CLASSES = ["maize/streak_virus", "maize/fall_armyworm", "cassava/mosaic", "cassava/green_mite",
                  "tomato/septoria_leaf_spot", "tomato/leaf_curl", "cashew/anthracnose", "cashew/healthy"]


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, timeout=900)
def pick_samples(seed: int = 1) -> dict[str, bytes]:
    """Quelques photos du jeu de TEST CCMT (jamais vues à l'entraînement), réduites à 512 px."""
    import io
    import random

    from PIL import Image

    src, dst = _unpack("ccmt")
    rows = [r for r in _read_csv(f"{VOL_PATH}/split.csv") if r["split"] == "test"]
    rnd = random.Random(seed)
    out = {}
    for cid in SAMPLE_CLASSES:
        r = rnd.choice([r for r in rows if r["label"] == cid])
        img = Image.open(r["path"].replace(src, dst, 1)).convert("RGB")
        img.thumbnail((512, 512))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        out[cid.replace("/", "__") + ".jpg"] = buf.getvalue()
    return out


@app.local_entrypoint()
def samples():
    from pathlib import Path

    d = Path("samples")
    d.mkdir(exist_ok=True)
    for name, data in pick_samples.remote().items():
        (d / name).write_bytes(data)
        print("écrit", d / name)
