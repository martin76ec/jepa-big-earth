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

import gc
import json
import shutil
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
    """Envuelve I-JEPA: imagen RGB -> vector de embedding (mean pooling).

    Memoria: carga con `low_cpu_mem_usage=True` (evita duplicar los pesos
    en RAM del host al cargar) y, en CUDA, usa fp16 (mitad de VRAM, mas
    rapido; el feature-extraction tolera fp16). En CPU se queda en fp32.
    """

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.cfg = cfg
        self.device = _device()
        self.dtype = torch.float16 if self.device == "cuda" else None
        self.processor = AutoImageProcessor.from_pretrained(cfg.jepa_model_id)
        kwargs = {"low_cpu_mem_usage": True}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(cfg.jepa_model_id, **kwargs)
        self.model.eval().to(self.device)
        self._torch = torch

    def encode(self, images_uint8: np.ndarray, ckpt_dir=None) -> np.ndarray:
        """images_uint8: [N,H,W,3] -> embeddings float32 [N,D].

        Si `ckpt_dir` se pasa, cada batch se persiste como `part_*.npy` y
        un batch ya persistido se salta: un kill a mitad del encode es
        recuperable (la corrida se reanuda donde quedo).
        """
        torch = self._torch
        bs = self.cfg.batch_size
        starts = list(range(0, len(images_uint8), bs))
        feats: list[np.ndarray] = []
        for k, i in enumerate(tqdm(starts, desc="jepa encode")):
            part = ckpt_dir / f"part_{k:06d}.npy" if ckpt_dir is not None else None
            if part is not None and part.exists():
                feats.append(np.load(part))
                continue
            batch = list(images_uint8[i : i + bs])
            inputs = {k2: v.to(self.device)
                      for k2, v in self.processor(images=batch,
                                                  return_tensors="pt").items()}
            if self.dtype is not None:
                inputs = {k2: (v.to(self.dtype) if torch.is_floating_point(v) else v)
                          for k2, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            # Mean pooling sobre los tokens de parche (de vuelta a fp32).
            emb = out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            if part is not None:
                np.save(part, emb)
            feats.append(emb)
            del inputs, out
        return np.concatenate(feats, axis=0)

    def release(self) -> None:
        """Libera el modelo y la cache CUDA antes de guardar / sklearn."""
        torch = self._torch
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def _emb_path(cfg: Config):
    return cfg.cache_dir / "embeddings.npz"


def _ckpt_dir(cfg: Config):
    return cfg.cache_dir / "emb_parts"


def embeddings_cached(cfg: Config) -> bool:
    return _emb_path(cfg).exists()


def _prepare_ckpt(cfg: Config, n: int):
    """Dir de checkpoints; invalida partes viejas si cambio el contexto.

    Las partes solo son reanudables si N, batch_size y modelo coinciden
    (mismo orden de batches). Si cambia algo, se descartan y se reinicia.
    """
    d = _ckpt_dir(cfg)
    meta_path = d / "meta.json"
    meta = {"n": int(n), "batch_size": cfg.batch_size,
            "model_id": cfg.jepa_model_id}
    if d.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))
    return d


def get_embeddings(cfg: Config, X_uint8: np.ndarray | None = None,
                   y: np.ndarray | None = None,
                   classes: list[str] | None = None, force: bool = False):
    """Calcula (o carga de cache) los embeddings JEPA del subconjunto.

    Si la cache existe y `force` es False, devuelve directamente sin
    necesitar el subconjunto de imagenes. El encode se hace con
    checkpointing en disco, asi que un kill por OOM es recuperable.
    """
    path = _emb_path(cfg)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"], list(data["classes"])

    if X_uint8 is None:
        raise ValueError("No hay cache de embeddings; se requiere el subconjunto.")

    cfg.make_dirs()
    ckpt = _prepare_ckpt(cfg, len(X_uint8))
    encoder = JepaEncoder(cfg)
    emb = encoder.encode(X_uint8, ckpt_dir=ckpt)
    encoder.release()  # libera modelo/VRAM antes de guardar
    np.savez_compressed(path, X=emb, y=y, classes=np.array(classes))
    shutil.rmtree(ckpt, ignore_errors=True)  # ya esta el .npz final
    return emb, y, classes
