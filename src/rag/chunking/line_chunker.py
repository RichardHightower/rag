from typing import List

from ..config import CHUNK_OVERLAP, CHUNK_SIZE
from ..model import Chunk, File
from .base_chunker import Chunker


class LineChunker(Chunker):
    """Chunker that splits text based on lines."""

    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Validate inputs
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    def chunk_text(self, file: File) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            file: File to split


        Returns:
            List[Chunk]: List of text chunks
        """

        lines = file.content.splitlines()
        if not lines:
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
