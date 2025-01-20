"""Mock embedder for testing."""

import random
from typing import List

from ..model import Chunk
from .base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder that returns random vectors."""

    def __init__(self, model_name: str = "mock", dimension: int = 1536):
        """Initialize mock embedder.

        Args:
            model_name: Name of the mock model
            dimension: Dimension of the embeddings
        """
        super().__init__(model_name, dimension)

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, texts: List[Chunk]) -> List[List[float]]:
        """Generate random embeddings for testing.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of random embeddings, one per text
        """
        return [[random.uniform(-1, 1) for _ in range(self.dimension)] for _ in texts]
