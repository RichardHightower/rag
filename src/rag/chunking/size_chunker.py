from typing import List

from ..config import CHUNK_OVERLAP, CHUNK_SIZE
from ..model import Chunk, File
from .base_chunker import Chunker


class SizeChunker(Chunker):
    """Chunker that splits text based on character size."""

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
        """Split text into overlapping chunks based on character size.

        Args:
            file: File to split

        Returns:
            List[Chunk]: List of text chunks
        """
        if file.content is None or (
            len(file.content) < 80 and file.content.strip() == ""
        ):
            return []

        content = file.content
        if not content:
            return [Chunk(target_size=self.chunk_size, content=content, index=0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            # If we're not at the end, try to find a good breaking point
            if end < len(content):
                # Look for the last occurrence of a space within the chunk
                last_space = content.rfind(" ", start, end)
                if last_space != -1 and last_space > start:
                    end = last_space

            chunk_content = content[start:end]
            chunks.append(
                Chunk(
                    target_size=self.chunk_size,
                    content=chunk_content,
                    index=chunk_index,
                )
            )

            # If we've reached the end, break
            if end == len(content):
                break

            # Move start position, accounting for overlap
            start = end - self.overlap
            chunk_index += 1

        return chunks
