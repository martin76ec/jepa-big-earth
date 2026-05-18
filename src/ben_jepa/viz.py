"""Visualizacion con t-SNE (1 pt del rubric).

Proyecta los embeddings JEPA a 2D y colorea cada punto por su clase
(tecnica supervisada -> color = etiqueta). Se aplica PCA previo a 50
componentes para acelerar y estabilizar t-SNE.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import Config


def tsne_plot(cfg: Config, X: np.ndarray, y: np.ndarray, classes: list[str]) -> None:
    cfg.make_dirs()
    n_pca = min(50, X.shape[1], X.shape[0])
    Xp = PCA(n_components=n_pca, random_state=cfg.random_state).fit_transform(X)
    emb2d = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=cfg.random_state,
    ).fit_transform(Xp)

    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, c in enumerate(classes):
        m = y == ci
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=8, alpha=0.6, label=c)
    ax.set_title("t-SNE de los embeddings JEPA (color = clase)")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "tsne.png", dpi=120)
    plt.close(fig)
