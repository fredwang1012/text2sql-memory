"""Hybrid search over knowledge items: BM25Plus (scaled by the best raw score) blended with cosine similarity.

Falls back to BM25 alone when the items have no stored vectors or no embedder is available.
"""
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Plus

from app.memory import Item, search_text

W_EMBED, W_BM25 = 0.6, 0.4

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def normalise(text: str) -> list[str]:
    """Lowercased words with a trailing "s" stripped from words longer than 3 letters (simple plural handling)."""
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in tokenize(text)]


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class Hit:
    item: Item
    score: float


class Index:
    def __init__(self, items: list[Item], vectors: list[np.ndarray | None] | None = None, embedder: Embedder | None = None):
        self.items = items
        usable = embedder is not None and vectors is not None and len(items) > 0 and all(v is not None for v in vectors)
        self.embeddings = np.vstack(vectors).astype(np.float32) if usable else None
        self.embedder = embedder if usable else None
        # BM25Plus rather than Okapi: Okapi's idf is 0 for a term in half the documents, so a small knowledge base
        # would score its own words 0 and never match.
        tokens = [tokenize(search_text(i)) for i in items]
        self._words = [set(t) for t in tokens]
        self._bm25 = BM25Plus(tokens) if items else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.embeddings is not None else "bm25"

    def scores(self, query: str) -> np.ndarray:
        n = len(self.items)
        terms = tokenize(query)
        bm = np.asarray(self._bm25.get_scores(terms), dtype=float) if self._bm25 and terms else np.zeros(n)
        hi = bm.max() if n else 0.0
        bm_norm = bm / hi if hi > 0 else np.zeros(n)
        if self.mode == "bm25" or not query.strip():
            return bm_norm
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        cos = self.embeddings @ (q / (np.linalg.norm(q) or 1.0))
        return W_EMBED * cos + W_BM25 * bm_norm

    def search(self, query: str, k: int) -> list[Hit]:
        """Top-k items, best first. BM25-only search returns only items that contain a query word."""
        if not self.items or not query.strip():
            return []
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        if self.mode == "bm25":
            # BM25Plus gives absent terms a constant positive score, so a score says nothing about a match
            terms = set(tokenize(query))
            order = [i for i in order if self._words[i] & terms]
        return [Hit(self.items[i], float(s[i])) for i in order]
