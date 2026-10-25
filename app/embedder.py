"""multilingual-e5-small query/passage embeddings. Imported lazily; retrieval falls back to BM25 if unavailable."""
import logging
import threading

import numpy as np

from app.config import settings

MODEL_NAME = "intfloat/multilingual-e5-small"
log = logging.getLogger(__name__)


class E5Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer  # heavy import: only when actually used
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode("query: " + text, normalize_embeddings=True)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=batch_size)


_loaded: E5Embedder | None = None
_failed = False
_lock = threading.Lock()


def get_embedder() -> E5Embedder | None:
    """The shared embedder, loaded on first use; None when embeddings are off or the model cannot be loaded."""
    global _loaded, _failed
    if settings.embed == "off":
        return None
    with _lock:
        if _failed:
            return None
        if _loaded is None:
            try:
                _loaded = E5Embedder()
            except ImportError:
                log.info("sentence-transformers not installed (uv sync --extra embed): BM25-only retrieval")
                _failed = True
            except Exception as e:  # e.g. the model cannot be downloaded offline
                log.warning("could not load %s (%s): BM25-only retrieval", MODEL_NAME, e)
                _failed = True
        return _loaded
