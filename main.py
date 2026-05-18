"""Punto de entrada CLI del proyecto.

Pipeline completo: BigEarthNet-S2 (RGB) -> embeddings JEPA -> clasificador
clasico (Regresion Logistica L2) vs SVM RBF, con EDA, optimizacion de
hiperparametros, comparacion estadistica y t-SNE.

Uso:
    python main.py all          # ejecuta todo en orden
    python main.py eda          # escaneo de etiquetas + EDA + subconjunto
    python main.py features     # extrae embeddings JEPA (cachea)
    python main.py tune         # optimizacion de hiperparametros
    python main.py compare      # comparacion estadistica de 2 tecnicas
    python main.py tsne         # visualizacion t-SNE

Nota de transparencia (rubric "Referencias"): la estructura de este
codigo fue generada con asistencia de un LLM (Claude Code). Prompt base:
"escribir un clasificador usando JEPA y encima un clasificador clasico
con las 5 clases mas frecuentes del dataset BigEarthNet-S2-v1.0, como
proyecto Python (no notebook), lo mas simple posible, cumpliendo el rubric".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.ben_jepa.config import CFG
from src.ben_jepa.data import (
    build_subset,
    load_cached_classes,
    scan_labels_cached,
    top_k_classes,
)
from src.ben_jepa.features import embeddings_cached, get_embeddings

# Forzar recomputo de embeddings JEPA aunque exista la cache. Lo setea
# main() desde --force-encode o la env var FORCE_ENCODE.
FORCE_ENCODE = False


def _resolve_classes():
    """Top-k desde cache si existe; si no, escanea el dataset."""
    cached = load_cached_classes(CFG)
    if cached:
        return cached
    label_counts, _ = scan_labels_cached(CFG)
    return top_k_classes(label_counts, CFG.top_k)


def _load_scan_and_subset():
    label_counts, cardinalities = scan_labels_cached(CFG)
    classes = top_k_classes(label_counts, CFG.top_k)
    X, y, classes = build_subset(CFG, classes)
    return label_counts, cardinalities, classes, X, y


def cmd_eda():
    from src.ben_jepa.eda import run_eda

    label_counts, cardinalities, classes, X, y = _load_scan_and_subset()
    summary = run_eda(CFG, label_counts, cardinalities, classes, X, y)
    print(json.dumps(summary, indent=2))


def cmd_features():
    emb, _, classes = _embeddings()
    print(f"Embeddings: {emb.shape}, clases: {classes}")


def _embeddings():
    """Embeddings JEPA, usando caches para no re-descargar ni recomputar."""
    if not FORCE_ENCODE and embeddings_cached(CFG):
        return get_embeddings(CFG)
    classes = _resolve_classes()
    X, y, classes = build_subset(CFG, classes)
    return get_embeddings(CFG, X, y, classes, force=FORCE_ENCODE)


def cmd_tune():
    from src.ben_jepa.tuning import tune_logreg

    emb, y, _ = _embeddings()
    print(json.dumps(tune_logreg(CFG, emb, y), indent=2))


def cmd_compare():
    from src.ben_jepa.compare import compare

    emb, y, _ = _embeddings()
    best_C = None
    tune_path = CFG.metrics_dir / "tuning.json"
    if tune_path.exists():
        best_C = json.loads(tune_path.read_text()).get("best_C")
    print(json.dumps(compare(CFG, emb, y, best_C=best_C), indent=2))


def cmd_tsne():
    from src.ben_jepa.viz import tsne_plot

    emb, y, classes = _embeddings()
    tsne_plot(CFG, emb, y, classes)
    print(f"t-SNE guardado en {CFG.fig_dir / 'tsne.png'}")


def cmd_all():
    from src.ben_jepa.compare import compare
    from src.ben_jepa.eda import run_eda
    from src.ben_jepa.features import get_embeddings
    from src.ben_jepa.tuning import tune_logreg
    from src.ben_jepa.viz import tsne_plot

    label_counts, cardinalities, classes, X, y = _load_scan_and_subset()
    run_eda(CFG, label_counts, cardinalities, classes, X, y)
    emb, y, classes = get_embeddings(CFG, X, y, classes, force=FORCE_ENCODE)
    # Libera el subconjunto de imagenes (~0.5 GB) y el escaneo antes de la
    # fase sklearn (paralela): reduce el pico de RAM que causaba el OOM.
    del X, label_counts, cardinalities
    import gc

    gc.collect()
    tune = tune_logreg(CFG, emb, y)
    cmp = compare(CFG, emb, y, best_C=tune["best_C"])
    tsne_plot(CFG, emb, y, classes)
    print(json.dumps({"tuning": tune, "comparison": cmp}, indent=2))


COMMANDS = {
    "eda": cmd_eda,
    "features": cmd_features,
    "tune": cmd_tune,
    "compare": cmd_compare,
    "tsne": cmd_tsne,
    "all": cmd_all,
}


def main():
    parser = argparse.ArgumentParser(description="BigEarthNet-S2 + JEPA + clasico")
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument(
        "--force-encode",
        action="store_true",
        help="recalcula los embeddings JEPA aunque exista la cache "
             "(util para verificar que el encode usa CUDA)",
    )
    args = parser.parse_args()
    env_force = os.environ.get("FORCE_ENCODE", "").lower() not in ("", "0", "false", "no")
    global FORCE_ENCODE
    FORCE_ENCODE = args.force_encode or env_force
    if not CFG.hf_token:
        print(
            "ADVERTENCIA: HF_TOKEN no esta seteado. El dataset es 'gated: auto' "
            "y probablemente falle la descarga. Ver .env.example.",
            file=sys.stderr,
        )
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
