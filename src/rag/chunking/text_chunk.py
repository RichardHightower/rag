"""Text chunking utilities."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents a chunk of text with its position in the original document."""

    content: str  # Changed from text to content to match the database model
    start: int
    end: int


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[TextChunk]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of TextChunk objects
    """

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        old_start = start  # track our position
        end = start + chunk_size

        # Find a split that doesn't break words, if possible
        if end < len(text):
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:
                end = start + chunk_size
        else:
            end = len(text)

        chunk_content = text[start:end].strip()
        chunks.append(TextChunk(content=chunk_content, start=start, end=end))

        # Calculate next start
        start = end - overlap
        if start < 0:
            start = 0

        # Safety: If start never moves forward, break to avoid infinite loop
        if start <= old_start:
            break

    return chunks

