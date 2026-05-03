from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > batch_size,
    )
    return np.asarray(vectors, dtype=np.float32).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text], batch_size=1)[0]
