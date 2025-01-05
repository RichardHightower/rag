"""Embeddings package for handling text embeddings from various sources."""

from .base_embedder import BaseEmbedder
from .mock_embedder import MockEmbedder

__all__ = ["BaseEmbedder", "MockEmbedder"]
