"""RAG + FAISS-powered job similarity search for Talent-Radar."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning("faiss-cpu not installed — RAG similarity search disabled")


class JobEmbedder:
    """Embeds job descriptions into fixed-size vectors for FAISS indexing."""

    VOCAB: dict[str, int] = {
        "python": 0, "machine learning": 1, "deep learning": 2, "sql": 3,
        "aws": 4, "kubernetes": 5, "react": 6, "typescript": 7, "go": 8,
        "rust": 9, "spark": 10, "pytorch": 11, "tensorflow": 12, "mlops": 13,
        "data engineering": 14, "senior": 15, "junior": 16, "remote": 17,
        "finance": 18, "healthcare": 19,
    }
    DIM = len(VOCAB)

    def embed(self, text: str) -> np.ndarray:
        """Return a bag-of-skills vector for the given job description text."""
        vec = np.zeros(self.DIM, dtype=np.float32)
        text_lower = text.lower()
        for term, idx in self.VOCAB.items():
            if term in text_lower:
                vec[idx] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class FAISSJobIndex:
    """FAISS flat L2 index for nearest-neighbour job retrieval."""

    def __init__(self) -> None:
        self.embedder = JobEmbedder()
        self.index: Any = None
        self.job_store: list[dict[str, Any]] = []

    def build(self, jobs: list[dict[str, Any]]) -> None:
        """Build the FAISS index from a list of job dicts (must have 'description' key)."""
        if not _HAS_FAISS:
            logger.warning("FAISS unavailable — index not built")
            return
        vectors = np.stack([self.embedder.embed(j.get("description", "")) for j in jobs])
        self.index = faiss.IndexFlatL2(self.embedder.DIM)
        self.index.add(vectors)
        self.job_store = jobs
        logger.info("FAISS index built with %d jobs", len(jobs))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k similar jobs for the query description."""
        if not _HAS_FAISS or self.index is None:
            return []
        vec = self.embedder.embed(query).reshape(1, -1)
        distances, indices = self.index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.job_store):
                results.append({**self.job_store[idx], "distance": float(dist)})
        return results
