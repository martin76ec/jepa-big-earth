"""Configuracion central del proyecto.

Todos los parametros estan en un solo lugar para mantener el codigo simple.
Se pueden sobreescribir con variables de entorno cuando aplique.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

# Carga el archivo .env (HF_TOKEN, etc.) si python-dotenv esta instalado.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:
    pass


@dataclass
class Config:
    # --- Dataset ---
    dataset_id: str = "danielz01/BigEarthNet-S2-v1.0"
    dataset_config: str = "s2-rgb"
    split: str = "train"

    # Cuantos ejemplos leer (solo etiquetas) para EDA + elegir el top-5.
    scan_size: int = 20_000
    # Numero de clases mas frecuentes a conservar.
    top_k: int = 5
    # Imagenes por clase en el subconjunto de modelado. Deliberadamente
    # pequeno (~5% de 700) para que el encode JEPA + sklearn sean rapidos:
    # 5 * 35 = 175 imagenes en total.
    per_class: int = 35
    # Tamano al que se redimensionan las imagenes antes de JEPA.
    image_size: int = 224

    # --- Extraccion de caracteristicas (JEPA / transfer learning) ---
    # I-JEPA ViT-H/14 preentrenado (Meta AI). Encoder congelado.
    jepa_model_id: str = "facebook/ijepa_vith14_1k"
    batch_size: int = 16

    # --- Seleccion de caracteristicas (filtro, opcional) ---
    # El rubric permite omitirla con transfer learning/imagenes. Se deja
    # como filtro opcional tipo SelectKBest. None = desactivada.
    select_k_best: int | None = None

    # --- Validacion cruzada ---
    cv_splits: int = 10
    cv_repeats: int = 3           # comparacion estadistica (10x3); subset
                                  # chico (175): mas repeticiones solo darian
                                  # falsa precision sobre los mismos datos
    tune_repeats: int = 3         # optimizacion de hiperparametros (mas barato)
    random_state: int = 42

    # Paralelismo de scikit-learn. n_jobs=-1 (un worker por core) agota la
    # RAM en servidores con muchos cores / limite de cgroup: se usa un tope
    # conservador, sobreescribible con la env var N_JOBS (p. ej. N_JOBS=2).
    n_jobs: int = field(default_factory=lambda: int(os.environ.get("N_JOBS", "4")))

    # --- Rutas de salida ---
    out_dir: Path = field(default=OUTPUTS)

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def eda_dir(self) -> Path:
        return self.out_dir / "eda"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / "metrics"

    @property
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def make_dirs(self) -> None:
        for d in (self.cache_dir, self.eda_dir, self.fig_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
