"""Extraccion de caracteristicas con JEPA (I-JEPA preentrenado, congelado).

Esto cubre el paso de "Feature Extraction" del rubric: el dataset es data
no estructurada (imagenes), asi que usamos un encoder JEPA preentrenado por
Meta AI como extractor de embeddings (transfer learning). El encoder NO se
entrena; sobre los embeddings se montan clasificadores clasicos.

Al usar transfer learning sobre imagenes, el rubric permite omitir el paso
de "Feature Selection"; de todos modos se deja como filtro opcional en
`models.py` (SelectKBest).
"""
from __future__ import annotations

import sys

import numpy as np
from tqdm import tqdm

from .config import Config


def _device():
    """Resuelve el device e imprime un reporte claro en stderr.

    Asi se sabe SIN DUDAS si JEPA corre en CUDA. Si cae a CPU se emite
    una advertencia bien visible (CPU es lento y agota la RAM del host).
    """
    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()
    if mps_ok:
        device = "mps"
    elif cuda_ok:
        device = "cuda"
    else:
        device = "cpu"

    print(
        f"[jepa] torch={torch.__version__} "
        f"torch.version.cuda={torch.version.cuda} "
        f"cuda_available={cuda_ok} -> device={device}",
        file=sys.stderr,
        flush=True,
    )
    if device == "cuda":
        print(
            f"[jepa] >>> USANDO CUDA: GPU 0 = "
            f"{torch.cuda.get_device_name(0)} <<<",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[jepa] *** ADVERTENCIA: NO se usa CUDA (device={device}). "
            f"JEPA correra lento y puede agotar la RAM del host. "
            f"Revisa la instalacion de torch (wheel +cpu?) y nvidia-smi. ***",
            file=sys.stderr,
            flush=True,
        )
    return device


class JepaEncoder:
    """Envuelve I-JEPA: imagen RGB -> vector de embedding (mean pooling)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.cfg = cfg
        self.device = _device()
        self.processor = AutoImageProcessor.from_pretrained(cfg.jepa_model_id)
        self.model = AutoModel.from_pretrained(cfg.jepa_model_id)
        self.model.eval().to(self.device)
        self._torch = torch

    def encode(self, images_uint8: np.ndarray) -> np.ndarray:
        """images_uint8: [N,H,W,3] -> embeddings float32 [N,D]."""
        torch = self._torch
        feats: list[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in tqdm(range(0, len(images_uint8), bs), desc="jepa encode"):
            batch = list(images_uint8[i : i + bs])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            # Mean pooling sobre los tokens de parche.
            emb = out.last_hidden_state.mean(dim=1)
            feats.append(emb.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


def _emb_path(cfg: Config):
    return cfg.cache_dir / "embeddings.npz"


def embeddings_cached(cfg: Config) -> bool:
    return _emb_path(cfg).exists()


def get_embeddings(cfg: Config, X_uint8: np.ndarray | None = None,
                   y: np.ndarray | None = None,
                   classes: list[str] | None = None, force: bool = False):
    """Calcula (o carga de cache) los embeddings JEPA del subconjunto.

    Si la cache existe y `force` es False, devuelve directamente sin
    necesitar el subconjunto de imagenes.
    """
    path = _emb_path(cfg)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"], list(data["classes"])

    if X_uint8 is None:
        raise ValueError("No hay cache de embeddings; se requiere el subconjunto.")

    encoder = JepaEncoder(cfg)
    emb = encoder.encode(X_uint8)
    cfg.make_dirs()
    np.savez_compressed(path, X=emb, y=y, classes=np.array(classes))
    return emb, y, classes
