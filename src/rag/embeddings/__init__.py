"""Embeddings package for handling text embeddings from various sources."""

from .base import Embedder
from .openai_embedder import OpenAIEmbedder
from .mock_embedder import MockEmbedder

__all__ = ['Embedder', 'OpenAIEmbedder', 'MockEmbedder']
