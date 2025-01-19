"""Tools to break files into chunks."""

from typing import Union

from ..model import File
from .base_chunker import Chunker
from .line_chunker import LineChunker
from .size_chunker import SizeChunker
from .word_chunker import WordChunker


# Add a debug function to help with troubleshooting
def debug_chunker(chunker: Chunker, text: Union[str, File]):
    if isinstance(text, str):
        file = File(
            name="debug.md",
            path="/debug/debug.md",
            crc="debug123",
            content=text,
            meta_data={},
        )
    else:
        file = text
    chunks = chunker.chunk_text(file)
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.content}")


__all__ = ["LineChunker", "WordChunker", "SizeChunker", "debug_chunker"]
