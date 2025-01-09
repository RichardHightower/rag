"""File and Chunk models using Pydantic."""

from typing import Dict, List

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


class File(BaseModel):
    """File model."""

    name: str = Field(..., min_length=1, max_length=80)
    path: str = Field(..., min_length=1, max_length=255)
    crc: str
    content: str
    meta_data: Dict[str, str] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get the actual size of the chunk content."""
        return len(self.content)


class Chunk(BaseModel):
    """Chunk model."""

    target_size: PositiveInt
    content: str
    index: NonNegativeInt

    @property
    def size(self) -> int:
        """Get the actual size of the chunk content."""
        return len(self.content)


class ChunkResult(BaseModel):
    """A single chunk result with similarity score."""

    score: float = Field(..., ge=0.0, le=1.0)
    chunk: Chunk


class ChunkResults(BaseModel):
    """Container for chunk search results with pagination info."""

    results: List[ChunkResult]
    total_count: NonNegativeInt
    page: PositiveInt
    page_size: PositiveInt

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total_count + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Check if there are more pages."""
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        """Check if there are previous pages."""
        return self.page > 1
