from typing import List, Optional

from ..config import Config
from ..model import Chunk, File
from .base_chunker import Chunker


class LineChunker(Chunker):
    """Chunker that splits text based on lines."""

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
        """Split text into overlapping chunks.

        Args:
            file: File to split


        Returns:
            List[Chunk]: List of text chunks
        """

        # Split text into lines
        if file.content is None:
            return []

        lines = file.content.splitlines()
        if not lines and file.content:
            return [Chunk(target_size=self.chunk_size, content=file.content, index=0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(lines):
            # Calculate end of current chunk
            end = min(start + self.chunk_size, len(lines))

            # Join lines for this chunk
            chunk_content = "\n".join(lines[start:end])
            chunks.append(
                Chunk(
                    target_size=self.chunk_size,
                    content=chunk_content,
                    index=chunk_index,
                )
            )

            # If we've reached the end, break
            if end == len(lines):
                break

            # Move start position, accounting for overlap
            start = end - self.overlap
            chunk_index += 1

        return chunks

    @classmethod
    def create(cls, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        config = Config(CHUNK_SIZE=chunk_size, CHUNK_OVERLAP=overlap)
        return LineChunker(config)
