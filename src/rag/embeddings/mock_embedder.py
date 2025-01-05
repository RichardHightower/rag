"""Mock embedder for testing."""

import random
from typing import List

from .base_embedder import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder that generates random embeddings for testing."""

    def __init__(self, dimension: int = 1536):
        """Initialize mock embedder.
        
        Args:
            dimension: Dimension of the embeddings to generate
        """
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of random embedding vectors
        """
        return [
            [random.uniform(-1, 1) for _ in range(self.dimension)]
            for _ in texts
        ]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents for compatibility.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of random embedding vectors
        """
        return self.embed_documents(texts)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        return self.dimension
