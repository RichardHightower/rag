"""Base chunker interface."""

from abc import ABC, abstractmethod
from typing import List

from ..model import Chunk, File


class Chunker(ABC):
    """Abstract base class for text chunking implementations."""

    @abstractmethod
    def chunk_text(self, file: File) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            file: File to split
            chunk_size: Number of lines per chunk
            overlap: Number of lines to overlap between chunks

        Returns:
            List[Chunk]: List of text chunks
        """
        pass
