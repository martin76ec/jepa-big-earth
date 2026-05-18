"""Comparacion estadistica de 2 tecnicas ML (3.5 pts del rubric).

Tecnica 1: Regresion Logistica Regularizada (obligatoria).
Tecnica 2: SVM con kernel RBF.

Se usa repeated k-fold cross-validation (por defecto 10 repeticiones x
10 folds) y un test de hipotesis pareado de Wilcoxon (signed-rank) sobre
las accuracies por fold. Ambos modelos se evaluan sobre EXACTAMENTE los
mismos splits (mismo objeto cv) para que la comparacion sea pareada.

H0: no hay diferencia de rendimiento entre ambas tecnicas.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from .config import Config
from .models import make_logreg, make_svm

ALPHA = 0.05


def compare(cfg: Config, X: np.ndarray, y: np.ndarray,
            best_C: float | None = None) -> dict:
    cfg.make_dirs()
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.cv_splits,
        n_repeats=cfg.cv_repeats,
        random_state=cfg.random_state,
    )

    logreg = make_logreg(cfg, C=best_C if best_C else 1.0)
    svm = make_svm(cfg)

    s_lr = cross_val_score(logreg, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=cfg.n_jobs)
    s_sv = cross_val_score(svm, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=cfg.n_jobs)

    stat, p = wilcoxon(s_lr, s_sv)
    if p < ALPHA:
        winner = "LogReg" if s_lr.mean() > s_sv.mean() else "SVM"
        verdict = (
            f"Se rechaza H0 (p={p:.4g} < {ALPHA}): diferencia significativa. "
            f"Mejor modelo: {winner}."
        )
    else:
        verdict = (
            f"No se rechaza H0 (p={p:.4g} >= {ALPHA}): "
            f"sin diferencia estadisticamente significativa."
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([s_lr, s_sv])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["LogReg L2", "SVM RBF"])
    ax.set_ylabel("Balanced accuracy por fold")
    ax.set_title(f"Comparacion ({cfg.cv_repeats}x{cfg.cv_splits} CV)\nWilcoxon p={p:.4g}")
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "model_comparison.png", dpi=120)
    plt.close(fig)

    result = {
        "cv": f"{cfg.cv_repeats} repeticiones x {cfg.cv_splits} folds",
        "logreg_bal_acc_mean": float(s_lr.mean()),
        "logreg_bal_acc_std": float(s_lr.std()),
        "svm_bal_acc_mean": float(s_sv.mean()),
        "svm_bal_acc_std": float(s_sv.std()),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "alpha": ALPHA,
        "verdict": verdict,
    }
    (cfg.metrics_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    return result
