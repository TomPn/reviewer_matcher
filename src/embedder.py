"""Sentence-transformer embedding wrapper — singleton, runs offline after first download."""

from __future__ import annotations
import logging
import os
from typing import Union
from pathlib import Path

log = logging.getLogger(__name__)

_model = None
_model_name: str = ""


def load_model(model_name_or_path: str, offline: bool = True):
    global _model, _model_name
    from sentence_transformers import SentenceTransformer

    model_path = Path(model_name_or_path)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(
            f"Local embedding model not found: {model_name_or_path}\n"
            f"Download the model once and place it in that folder."
        )

    log.info(f"Loading embedding model from local path: {model_path}")
    _model = SentenceTransformer(str(model_path), local_files_only=True)
    _model_name = model_name_or_path
    return _model

def _ensure_loaded():
    if _model is None:
        raise RuntimeError("Call embedder.load_model(name) before encoding.")


def embed(texts: Union[str, list[str]], batch_size: int = 64, show_progress: bool = False) -> list[list[float]]:
    _ensure_loaded()
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    vectors = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise: cosine sim = dot product
    )
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed(text)[0]


def dimension() -> int:
    _ensure_loaded()
    return _model.get_sentence_embedding_dimension()