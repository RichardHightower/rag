"""OpenAI embedder implementation."""

import os
from typing import List, Optional

from ..config import Config
from ..model import Chunk
from .base import Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI embedder implementation."""

    def __init__(
        self,
        config: Optional[Config] = None,
        batch_size: int = 16,
    ):
        """Initialize OpenAI embedder.

        Args:
           config: Configuration

        Raises:
            ValueError: If no API key is provided or if config is None
        """
        if config is None:
            config = Config()

        super().__init__(config.OPENAI_TEXT_EMBED_MODEL, config.EMBEDDING_DIM)
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be provided")
        import openai

        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.batch_size = batch_size

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API.

        Args:
            chunks: List of text chunks to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = [chunk.content for chunk in chunks[i : i + self.batch_size]]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size=16,
    ):
        config = Config(
            OPENAI_API_KEY=api_key,
            OPENAI_TEXT_EMBED_MODEL=model_name,
            EMBEDDING_DIM=dimension,
        )
        return OpenAIEmbedder(config, batch_size=batch_size)
