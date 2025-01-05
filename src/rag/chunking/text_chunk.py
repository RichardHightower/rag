"""Text chunking utilities."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents a chunk of text with its position in the original document."""

    content: str  # Changed from text to content to match the database model
    start: int
    end: int


def find_word_boundary(text: str, pos: int, direction: int = 1) -> int:
    """Find the nearest word boundary in the given direction.
    
    Args:
        text: Text to search in
        pos: Starting position
        direction: 1 for forward, -1 for backward
        
    Returns:
        Position of the nearest word boundary
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    # Handle edge cases
    if pos <= 0:
        return 0
    if pos >= len(text):
        return len(text)

    # Normalize position to be within bounds
    pos = min(max(0, pos), len(text))

    # When going backward, look for space before position
    # When going forward, look for space at or after position
    current = pos
    while 0 <= current < len(text):
        # When going backward, we want to find a space and return the position after it
        if direction == -1:
            if current == 0:
                return 0
            if text[current - 1].isspace():
                return current
            current -= 1
        # When going forward, we want to find a space and return that position
        else:
            if text[current].isspace():
                return current + 1
            if current == len(text) - 1:
                return len(text)
            current += 1

    # If no boundary found, return edge
    return len(text) if direction == 1 else 0


def split_text_into_chunks(
    text: str, chunk_size: int = 1000, overlap: int = 50
) -> List[TextChunk]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split into chunks
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of TextChunk objects

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size
    """
    # Input validation
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    # Handle empty text
    if not text:
        return []

    # If text is smaller than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [TextChunk(content=text, start=0, end=len(text))]

    chunks = []
    start = 0

    while start < len(text):
        # Find end position for this chunk
        target_end = min(start + chunk_size, len(text))
        
        # If this would be the last chunk, just go to the end
        if target_end == len(text):
            end = len(text)
        else:
            # Try to find a word boundary at or before target
            end = find_word_boundary(text, target_end, direction=1)
            # If no good boundary found or it would make chunk too small,
            # use exact size
            if end <= start + chunk_size // 2:
                end = target_end

        # Create chunk
        chunk = TextChunk(
            content=text[start:end],
            start=start,
            end=end
        )
        chunks.append(chunk)

        if end == len(text):
            break

        # Calculate next start position for exact overlap
        next_start = end - overlap

        # If remaining text would create a tiny final chunk, extend current chunk
        remaining = len(text) - next_start
        if remaining > 0 and remaining <= overlap:
            chunks[-1] = TextChunk(
                content=text[chunks[-1].start:len(text)],
                start=chunks[-1].start,
                end=len(text)
            )
            break

        # Find word boundary for next start to avoid splitting words
        start = find_word_boundary(text, next_start, direction=1)
        if start >= end - overlap // 2:  # If boundary would create too much overlap
            start = next_start  # Use exact overlap position

    return chunks
