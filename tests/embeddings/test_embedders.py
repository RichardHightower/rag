"""Test embedders."""

import pytest

from vector_rag.embeddings import MockEmbedder


def test_mock_embedder():
    """Test mock embedder."""
    embedder = MockEmbedder(dimension=4)
    texts = ["Hello, world!", "Another test"]

    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 4
    assert all(-1 <= x <= 1 for x in embeddings[0])
