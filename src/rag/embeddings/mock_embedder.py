"""Mock embedder for testing."""

import random
from typing import List
from .base_embedder import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder that returns random vectors."""

    def __init__(self, dimension: int = 1536):
        """Initialize mock embedder.

        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return [[random.random() for _ in range(self.dimension)] for _ in texts]
