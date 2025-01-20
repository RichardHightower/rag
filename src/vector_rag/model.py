"""File and Chunk models using Pydantic."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

AnyValue = Union[str, int, float, List[str], Dict[str, Any]]
AnyValue1 = Union[str, int, float, List[str], Dict[str, AnyValue]]
AnyValue2 = Union[str, int, float, List[str], Dict[str, AnyValue1]]
AnyValue3 = Union[str, int, float, List[str], Dict[str, AnyValue2]]
MetaDataDict = Dict[str, AnyValue3]


class Project(BaseModel):
    """File model."""

    id: NonNegativeInt
    name: str = Field(..., min_length=1, max_length=80)
    description: Optional[str] = Field(..., min_length=1, max_length=255)


class File(BaseModel):
    """File model."""

    id: Optional[NonNegativeInt] = None
    name: str = Field(..., min_length=1, max_length=80)
    path: str = Field(..., min_length=1, max_length=255)
    crc: str
    content: Optional[str] = None
    meta_data: Dict[str, str] = Field(default_factory=dict)
    file_size: Optional[NonNegativeInt] = None

    @property
    def size(self) -> int:
        """Get the actual size of the chunk content."""
        if self.file_size is not None:
            return self.file_size
        else:
            if self.content is None:
                return 0
            else:
                return len(self.content)


class Chunk(BaseModel):
    """Chunk model."""

    target_size: PositiveInt
    content: str
    index: NonNegativeInt
    meta_data: MetaDataDict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
