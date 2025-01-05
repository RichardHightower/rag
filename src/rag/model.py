"""File and Chunk models using Pydantic."""

from typing import Dict

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
