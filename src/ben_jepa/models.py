"""Pipelines de scikit-learn para los clasificadores clasicos.

Tecnica 1 (obligatoria por el rubric): Regresion Logistica Regularizada (L2).
Tecnica 2 (para la comparacion estadistica): SVM con kernel RBF.

Ambas viven dentro de un sklearn.Pipeline (StandardScaler -> [filtro opc.]
-> clasificador), cubriendo el punto "Pipelines" del rubric.
"""
from __future__ import annotations

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config


def _maybe_select(cfg: Config) -> list:
    """Filtro opcional de seleccion de caracteristicas (tipo filter)."""
    if cfg.select_k_best is None:
        return []
    return [("select", SelectKBest(score_func=f_classif, k=cfg.select_k_best))]


def make_logreg(cfg: Config, C: float = 1.0) -> Pipeline:
    """Regresion Logistica con regularizacion L2 (lambda = 1/C)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            *_maybe_select(cfg),
            (
                "clf",
                LogisticRegression(
                    # L2 puro: l1_ratio=0 (en sklearn>=1.8 'penalty' esta
                    # deprecado; con penalty por defecto + l1_ratio=0 da
                    # L2 y es retrocompatible con sklearn>=1.4).
                    l1_ratio=0,
                    C=C,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def make_svm(cfg: Config, C: float = 1.0, gamma: str | float = "scale") -> Pipeline:
    """SVM con kernel RBF (segunda tecnica para la comparacion)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            *_maybe_select(cfg),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )
