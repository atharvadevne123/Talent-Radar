"""Tests for RAG job similarity search module."""
from __future__ import annotations

import numpy as np
import pytest

from app.rag import FAISSJobIndex, JobEmbedder


class TestJobEmbedder:
    def test_embed_returns_numpy_array(self):
        embedder = JobEmbedder()
        vec = embedder.embed("python machine learning")
        assert isinstance(vec, np.ndarray)

    def test_embed_known_term_nonzero(self):
        embedder = JobEmbedder()
        vec = embedder.embed("python")
        assert vec[embedder.VOCAB["python"]] > 0

    def test_embed_empty_string_all_zero(self):
        embedder = JobEmbedder()
        vec = embedder.embed("")
        assert vec.sum() == 0.0

    def test_embed_normalized(self):
        embedder = JobEmbedder()
        vec = embedder.embed("python machine learning sql aws")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6 or norm == 0.0

    def test_embed_dim_matches_vocab(self):
        embedder = JobEmbedder()
        vec = embedder.embed("python")
        assert len(vec) == embedder.DIM


class TestFAISSJobIndex:
    @pytest.fixture()
    def sample_jobs(self):
        return [
            {"title": "Python Engineer", "description": "python machine learning aws", "salary": 120_000},
            {"title": "Frontend Dev", "description": "react typescript", "salary": 90_000},
            {"title": "MLOps Engineer", "description": "python kubernetes mlops pytorch", "salary": 150_000},
        ]

    def test_build_and_search(self, sample_jobs):
        index = FAISSJobIndex()
        index.build(sample_jobs)
        if index.index is None:
            pytest.skip("FAISS not installed")
        results = index.search("python machine learning", top_k=2)
        assert len(results) <= 2

    def test_search_without_build_returns_empty(self):
        index = FAISSJobIndex()
        results = index.search("python", top_k=3)
        assert results == []

    def test_search_returns_distance_key(self, sample_jobs):
        index = FAISSJobIndex()
        index.build(sample_jobs)
        if index.index is None:
            pytest.skip("FAISS not installed")
        results = index.search("python", top_k=1)
        if results:
            assert "distance" in results[0]

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_search_respects_top_k(self, sample_jobs, top_k):
        index = FAISSJobIndex()
        index.build(sample_jobs)
        if index.index is None:
            pytest.skip("FAISS not installed")
        results = index.search("machine learning", top_k=top_k)
        assert len(results) <= top_k
