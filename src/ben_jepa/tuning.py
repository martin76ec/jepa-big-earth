"""Optimizacion de hiperparametros (3.5 pts del rubric).

Para la Regresion Logistica Regularizada se optimiza C (= 1/lambda) con
GridSearchCV sobre RepeatedStratifiedKFold. Se generan:
  - curva de validacion (lambda vs balanced accuracy, train/val)
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
)

from .config import Config
from .models import make_logreg

# lambda = 1/C. Rango acotado a la region que converge y es informativa:
# C grande (poca regularizacion) no converge en 2000 iters de lbfgs y,
# segun los resultados, da peor balanced accuracy -> se descarta.
C_GRID = np.logspace(-3, 1, 9)


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
        f"1/2 GridSearchCV: {len(C_GRID)} C x "
        f"{cfg.cv_splits}x{cfg.tune_repeats} folds"
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": C_GRID},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=True,
        return_train_score=True,  # para la curva de validacion (sin reejecutar)
        verbose=2,
    )
    grid.fit(X, y)
    best_C = grid.best_params_["clf__C"]

    # --- Curva de validacion (lambda = 1/C) ---
    # Derivada de grid.cv_results_: GridSearchCV YA evaluo train/val por
    # cada C sobre el MISMO CV, asi que no hace falta reejecutar
    # validation_curve (eran ~270 fits identicos). Resultado: identico.
    res = grid.cv_results_
    order = np.argsort(np.asarray(res["param_clf__C"], dtype=float))
    train_mean = res["mean_train_score"][order]
    train_std = res["std_train_score"][order]
    val_mean = res["mean_test_score"][order]
    val_std = res["std_test_score"][order]
    lambdas = 1.0 / C_GRID
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lambdas, train_mean, "o-", label="Entrenamiento")
    ax.fill_between(lambdas, train_mean - train_std,
                    train_mean + train_std, alpha=0.15)
    ax.semilogx(lambdas, val_mean, "o-", label="Validacion")
    ax.fill_between(lambdas, val_mean - val_std,
                    val_mean + val_std, alpha=0.15)
    ax.axvline(1.0 / best_C, color="red", ls="--", label=f"lambda* = {1.0/best_C:.3g}")
    ax.set_xlabel("lambda (1/C)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Curva de validacion - Regresion Logistica L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "validation_curve.png", dpi=120)
    plt.close(fig)

    # --- Curva de aprendizaje (con el mejor C) ---
    _phase(f"2/2 learning_curve (best C={best_C:.3g})")
    # shuffle=True: sin esto learning_curve toma los PRIMEROS k indices
    # del fold (agrupados por clase) y los train_sizes chicos quedan de
    # una sola clase -> "needs samples of at least 2 classes".
    sizes, tr, va = learning_curve(
        make_logreg(cfg, C=best_C), X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="balanced_accuracy", cv=cv, n_jobs=cfg.n_jobs, verbose=1,
        shuffle=True, random_state=cfg.random_state,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tr.mean(1), "o-", label="Entrenamiento")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), alpha=0.15)
    ax.plot(sizes, va.mean(1), "o-", label="Validacion")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.15)
    ax.set_xlabel("Tamano de entrenamiento")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Curva de aprendizaje - Regresion Logistica L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "learning_curve.png", dpi=120)
    plt.close(fig)

    result = {
        "best_C": float(best_C),
        "best_lambda": float(1.0 / best_C),
        "best_cv_bal_acc": float(grid.best_score_),
        "C_grid": [float(c) for c in C_GRID],
        "val_bal_acc_mean": [float(v) for v in val_mean],
    }
    (cfg.metrics_dir / "tuning.json").write_text(json.dumps(result, indent=2))
    return result
