"""Embeddings package for handling text embeddings from various sources."""

from .base import Embedder
from .mock_embedder import MockEmbedder
from .openai_embedder import OpenAIEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "MockEmbedder"]
