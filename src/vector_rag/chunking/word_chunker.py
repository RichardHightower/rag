from typing import List, Optional

from ..config import Config
from ..model import Chunk, File
from .base_chunker import Chunker


class WordChunker(Chunker):
    """Chunker that splits text based on words."""

    def __init__(self, config: Config = Config()):
        self.chunk_size = config.CHUNK_SIZE
        self.overlap = config.CHUNK_OVERLAP
        # Validate inputs
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    def chunk_text(self, file: File) -> List[Chunk]:
        """Split text into overlapping chunks based on words.

        Args:
            file: File to split

        Returns:
            List[Chunk]: List of text chunks
        """
        if file.content is None:
            return []

        words = file.content.split()
        if not words:
            return [Chunk(target_size=self.chunk_size, content=file.content, index=0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))

            # Join words for this chunk
            chunk_content = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    target_size=self.chunk_size,
                    content=chunk_content,
                    index=chunk_index,
                )
            )

            # If we've reached the end, break
            if end == len(words):
                break

            # Move start position, accounting for overlap
            start = end - self.overlap
            chunk_index += 1

        return chunks

    @classmethod
    def create(cls, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        config = Config(CHUNK_SIZE=chunk_size, CHUNK_OVERLAP=overlap)
        return WordChunker(config)
