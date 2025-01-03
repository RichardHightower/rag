"""OpenAI embedder implementation."""

from typing import List
import openai
from .base import Embedder
from ..config import OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_DIM


class OpenAIEmbedder(Embedder):
    """OpenAI embedder implementation."""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        dimension: int = EMBEDDING_DIM,
        api_key: str = OPENAI_API_KEY,
        batch_size: int = 16,
    ):
        """Initialize OpenAI embedder.

        Args:
            model_name: Name of the OpenAI model to use
            dimension: Dimension of the embeddings
            api_key: OpenAI API key
            batch_size: Number of texts to embed in one batch
        """
        super().__init__(model_name, dimension)
        if not api_key:
            raise ValueError("OpenAI API key must be provided")
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = batch_size

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings
