"""Base embedder interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Base class for embedding generators."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings generated by this embedder.

        Returns:
            Integer dimension of the embedding vectors
        """
        pass
