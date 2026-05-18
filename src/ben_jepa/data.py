"""Carga del dataset BigEarthNet-S2 (config s2-rgb) desde Hugging Face.

Estrategia, pensada para correr en una Mac sin GPU CUDA:

1. `scan_labels`  -> pasada en streaming leyendo SOLO la columna `labels`
   (no decodifica imagenes, es rapida). Sirve para el EDA y para elegir
   el top-k de clases mas frecuentes.
2. `build_subset` -> segunda pasada en streaming que recolecta, por cada
   clase del top-k, hasta `per_class` imagenes que sean de etiqueta unica
   dentro del top-k (problema de clasificacion multiclase simple).
   El resultado se cachea en disco como .npz para no re-descargar.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config


def _load_stream(cfg: Config):
    from datasets import load_dataset

    return load_dataset(
        cfg.dataset_id,
        cfg.dataset_config,
        split=cfg.split,
        streaming=True,
        token=cfg.hf_token,
    )


def scan_labels(cfg: Config) -> tuple[Counter, list[int]]:
    """Cuenta frecuencia de etiquetas y cardinalidad por parche.

    Devuelve (Counter de etiquetas, lista con #etiquetas por parche).
    """
    ds = _load_stream(cfg).select_columns(["labels"])
    label_counts: Counter = Counter()
    cardinalities: list[int] = []
    for i, ex in enumerate(tqdm(ds, total=cfg.scan_size, desc="scan labels")):
        if i >= cfg.scan_size:
            break
        labels = ex["labels"]
        label_counts.update(labels)
        cardinalities.append(len(labels))
    return label_counts, cardinalities


def _scan_path(cfg: Config):
    return cfg.cache_dir / "scan.json"


def scan_labels_cached(cfg: Config, force: bool = False) -> tuple[Counter, list[int]]:
    """`scan_labels` con cache en disco (clave: `scan_size`).

    `scan_labels` no se cacheaba: cada corrida de `eda`/`all` repetia la
    pasada de streaming de `scan_size` ejemplos. Aqui persistimos el
    Counter de etiquetas y la lista de cardinalidades en `scan.json`. Si
    cambia `scan_size` (o `force=True`), la cache se invalida y se reescanea.
    """
    path = _scan_path(cfg)
    if path.exists() and not force:
        data = json.loads(path.read_text())
        if data.get("scan_size") == cfg.scan_size:
            return Counter(data["label_counts"]), list(data["cardinalities"])

    label_counts, cardinalities = scan_labels(cfg)
    cfg.make_dirs()
    path.write_text(
        json.dumps(
            {
                "scan_size": cfg.scan_size,
                "label_counts": dict(label_counts),
                "cardinalities": cardinalities,
            }
        )
    )
    return label_counts, cardinalities


def top_k_classes(label_counts: Counter, k: int) -> list[str]:
    return [name for name, _ in label_counts.most_common(k)]


def _subset_paths(cfg: Config):
    return cfg.cache_dir / "subset.npz", cfg.cache_dir / "classes.json"


def load_cached_classes(cfg: Config) -> list[str] | None:
    """Devuelve el top-k cacheado (si existe) para evitar re-escanear."""
    _, cls_path = _subset_paths(cfg)
    if cls_path.exists():
        return json.loads(cls_path.read_text())
    return None


# Logica de construccion del subconjunto. Bump => invalida la cache
# subset.npz (junto con per_class) para no servir un subconjunto viejo.
BUILD_VERSION = 2


def build_subset(cfg: Config, classes: list[str], force: bool = False):
    """Construye (o carga de cache) el subconjunto multiclase balanceado.

    BigEarthNet es multi-etiqueta (~3 etiquetas/parche): exigir parches
    con UNA sola etiqueta del top-5 dejaba a las clases minoritarias casi
    sin muestras (subconjunto degenerado de una sola clase, p. ej. todo
    clase 0). En su lugar se toma cualquier parche con >=1 etiqueta del
    top-5 y se le asigna, de entre esas, la clase menos representada hasta
    el momento (balanceo greedy, determinista por orden de frecuencia).

    Devuelve (X uint8 [N,H,W,3], y int [N], classes list[str]).
    `classes` define el orden de los indices de clase.
    """
    npz_path, cls_path = _subset_paths(cfg)
    if npz_path.exists() and cls_path.exists() and not force:
        cached_classes = json.loads(cls_path.read_text())
        data = np.load(npz_path)
        if (cached_classes == classes
                and "per_class" in data.files
                and int(data["per_class"]) == cfg.per_class
                and "build_version" in data.files
                and int(data["build_version"]) == BUILD_VERSION):
            return data["X"], data["y"], cached_classes

    class_to_idx = {c: i for i, c in enumerate(classes)}
    top = set(classes)
    target = cfg.per_class
    counts = {c: 0 for c in classes}
    # Tope de streaming: si una clase es demasiado rara para llegar a
    # `target`, no recorrer indefinidamente el dataset (descarga lenta).
    cap = max(cfg.scan_size, target * len(classes) * 10)

    ds = _load_stream(cfg).select_columns(["img", "labels"])
    X: list[np.ndarray] = []
    y: list[int] = []
    pbar = tqdm(total=target * len(classes), desc="build subset")
    for n_seen, ex in enumerate(ds):
        if n_seen >= cap:
            break
        present = top.intersection(ex["labels"])
        if not present:
            continue
        # Asigna la clase top-5 presente menos llena (balanceo greedy);
        # orden por frecuencia para que el desempate sea determinista.
        cls = min(sorted(present, key=lambda c: class_to_idx[c]),
                  key=lambda c: counts[c])
        if counts[cls] >= target:
            continue
        img: Image.Image = ex["img"].convert("RGB").resize(
            (cfg.image_size, cfg.image_size), Image.BILINEAR
        )
        X.append(np.asarray(img, dtype=np.uint8))
        y.append(class_to_idx[cls])
        counts[cls] += 1
        pbar.update(1)
        if all(v >= target for v in counts.values()):
            break
    pbar.close()

    min_needed = max(2, cfg.cv_splits)
    short = {c: n for c, n in counts.items() if n < min_needed}
    if short:
        raise ValueError(
            f"Subconjunto degenerado: clases con muy pocas muestras {short} "
            f"(se necesitan >= {min_needed} por clase para CV de "
            f"{cfg.cv_splits} folds). counts={counts}. Sube scan_size o "
            f"baja cv_splits/per_class."
        )

    X_arr = np.stack(X)
    y_arr = np.asarray(y, dtype=np.int64)
    cfg.make_dirs()
    np.savez_compressed(npz_path, X=X_arr, y=y_arr,
                        per_class=cfg.per_class, build_version=BUILD_VERSION)
    cls_path.write_text(json.dumps(classes))
    return X_arr, y_arr, classes
