"""Sentence-transformer embedding wrapper — singleton, runs offline after first download."""

from __future__ import annotations
import logging
import os
from typing import Union

log = logging.getLogger(__name__)

_model = None
_model_name: str = ""


def load_model(model_name: str, offline: bool = False) -> None:
    global _model, _model_name
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("Run: pip install sentence-transformers") from e

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    log.info(f"Loading embedding model: '{model_name}'")
    _model = SentenceTransformer(model_name)
    _model_name = model_name
    log.info(f"Model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")


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