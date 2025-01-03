"""Text chunking utilities."""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Number of lines per chunk
        overlap: Number of lines to overlap between chunks

    Returns:
        List[str]: List of text chunks
    """
    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    lines = text.splitlines()
    if not lines:
        return [text]  # Return original text if empty or just whitespace
        
    chunks = []
    start = 0

    while start < len(lines):
        # Calculate end of current chunk
        end = min(start + chunk_size, len(lines))
        
        # Join lines for this chunk
        chunk = '\n'.join(lines[start:end])
        chunks.append(chunk)
        
        # If we've reached the end, break
        if end == len(lines):
            break
            
        # Move start position, accounting for overlap
        start = end - overlap

    return chunks
