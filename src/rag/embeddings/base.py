"""Base embedder interface."""

from abc import ABC, abstractmethod
from typing import List

from rag.model import Chunk


class Embedder(ABC):
    """Base class for all embedders."""

    def __init__(self, model_name: str, dimension: int):
        """Initialize embedder with model name and dimension.

        Args:
            model_name: Name of the model to use
            dimension: Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[Chunk]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        pass
