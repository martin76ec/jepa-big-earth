"""Analisis Exploratorio de Datos (EDA) del rubric.

Genera, a partir de la pasada de escaneo de etiquetas y del subconjunto:
  - distribucion de frecuencia de etiquetas (top 20, top-k resaltado)
  - histograma de cardinalidad (etiquetas por parche, dataset multi-etiqueta)
  - tamano del subconjunto multiclase por clase
  - grilla de imagenes de ejemplo por clase
  - eda_summary.json con estadisticas
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


def run_eda(cfg: Config, label_counts: Counter, cardinalities: list[int],
            classes: list[str], X: np.ndarray, y: np.ndarray) -> dict:
    cfg.make_dirs()

    # 1. Frecuencia de etiquetas (top 20).
    common = label_counts.most_common(20)
    names = [n for n, _ in common]
    vals = [v for _, v in common]
    colors = ["#d62728" if n in classes else "#1f77b4" for n in names]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names))[::-1], vals, color=colors)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"Frecuencia (muestra de {cfg.scan_size} parches)")
    ax.set_title("Distribucion de etiquetas (rojo = top-5 elegido)")
    fig.tight_layout()
    fig.savefig(cfg.eda_dir / "label_frequency.png", dpi=120)
    plt.close(fig)

    # 2. Cardinalidad (multi-etiqueta).
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cardinalities, bins=range(1, max(cardinalities) + 2),
            align="left", color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Etiquetas por parche")
    ax.set_ylabel("Cantidad de parches")
    ax.set_title("Cardinalidad de etiquetas (dataset multi-etiqueta)")
    fig.tight_layout()
    fig.savefig(cfg.eda_dir / "label_cardinality.png", dpi=120)
    plt.close(fig)

    # 3. Balance del subconjunto multiclase.
    counts = np.bincount(y, minlength=len(classes))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(classes)), counts, color="#1f77b4")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Imagenes (etiqueta unica)")
    ax.set_title("Subconjunto de modelado por clase")
    fig.tight_layout()
    fig.savefig(cfg.eda_dir / "subset_balance.png", dpi=120)
    plt.close(fig)

    # 4. Grilla de ejemplos por clase.
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for ci, c in enumerate(classes):
        idx = np.where(y == ci)[0][0]
        axes[ci].imshow(X[idx])
        axes[ci].set_title(c, fontsize=8)
        axes[ci].axis("off")
    fig.suptitle("Ejemplos por clase (Sentinel-2 RGB)")
    fig.tight_layout()
    fig.savefig(cfg.eda_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    summary = {
        "scanned_patches": cfg.scan_size,
        "unique_labels": len(label_counts),
        "top_k_classes": classes,
        "top_k_frequencies": {c: label_counts[c] for c in classes},
        "mean_labels_per_patch": float(np.mean(cardinalities)),
        "subset_size": int(len(y)),
        "subset_per_class": {c: int(counts[i]) for i, c in enumerate(classes)},
    }
    (cfg.eda_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
