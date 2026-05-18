"""Optimizacion de hiperparametros (3.5 pts del rubric).

Para la Regresion Logistica Regularizada se optimiza C (= 1/lambda) con
GridSearchCV sobre RepeatedStratifiedKFold. Se generan:
  - curva de validacion (lambda vs accuracy, train/val)
  - curva de aprendizaje

Opcional/extra: se podria sustituir GridSearchCV por Optuna; se deja
GridSearchCV por simplicidad y reproducibilidad.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    learning_curve,
    validation_curve,
)

from .config import Config
from .models import make_logreg

C_GRID = np.logspace(-3, 3, 13)  # lambda = 1/C


def _phase(msg: str) -> None:
    """Banner de progreso a stderr (stdout queda limpio para el JSON)."""
    print(f"[tune] {msg}", file=sys.stderr, flush=True)


def tune_logreg(cfg: Config, X: np.ndarray, y: np.ndarray) -> dict:
    cfg.make_dirs()
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.cv_splits,
        n_repeats=cfg.tune_repeats,
        random_state=cfg.random_state,
    )
    pipe = make_logreg(cfg)

    _phase(
        f"1/3 GridSearchCV: {len(C_GRID)} C x "
        f"{cfg.cv_splits}x{cfg.tune_repeats} folds"
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": C_GRID},
        scoring="accuracy",
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=True,
        verbose=2,
    )
    grid.fit(X, y)
    best_C = grid.best_params_["clf__C"]

    # --- Curva de validacion (lambda = 1/C) ---
    _phase("2/3 validation_curve")
    train_sc, val_sc = validation_curve(
        make_logreg(cfg), X, y,
        param_name="clf__C", param_range=C_GRID,
        scoring="accuracy", cv=cv, n_jobs=cfg.n_jobs, verbose=1,
    )
    lambdas = 1.0 / C_GRID
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lambdas, train_sc.mean(1), "o-", label="Entrenamiento")
    ax.fill_between(lambdas, train_sc.mean(1) - train_sc.std(1),
                    train_sc.mean(1) + train_sc.std(1), alpha=0.15)
    ax.semilogx(lambdas, val_sc.mean(1), "o-", label="Validacion")
    ax.fill_between(lambdas, val_sc.mean(1) - val_sc.std(1),
                    val_sc.mean(1) + val_sc.std(1), alpha=0.15)
    ax.axvline(1.0 / best_C, color="red", ls="--", label=f"lambda* = {1.0/best_C:.3g}")
    ax.set_xlabel("lambda (1/C)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Curva de validacion - Regresion Logistica L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "validation_curve.png", dpi=120)
    plt.close(fig)

    # --- Curva de aprendizaje (con el mejor C) ---
    _phase(f"3/3 learning_curve (best C={best_C:.3g})")
    sizes, tr, va = learning_curve(
        make_logreg(cfg, C=best_C), X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy", cv=cv, n_jobs=cfg.n_jobs, verbose=1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tr.mean(1), "o-", label="Entrenamiento")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), alpha=0.15)
    ax.plot(sizes, va.mean(1), "o-", label="Validacion")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.15)
    ax.set_xlabel("Tamano de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title("Curva de aprendizaje - Regresion Logistica L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "learning_curve.png", dpi=120)
    plt.close(fig)

    result = {
        "best_C": float(best_C),
        "best_lambda": float(1.0 / best_C),
        "best_cv_accuracy": float(grid.best_score_),
        "C_grid": [float(c) for c in C_GRID],
        "val_accuracy_mean": [float(v) for v in val_sc.mean(1)],
    }
    (cfg.metrics_dir / "tuning.json").write_text(json.dumps(result, indent=2))
    return result
