from abc import ABC, abstractmethod
from typing import List, Optional

from vector_rag.model import ChunkResults, File, Project

from .db_model import FileDB, ProjectDB


class FileHandler(ABC):
    """Abstract base class for file handling operations."""

    @abstractmethod
    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project."""
        pass

    @abstractmethod
    def add_file(self, project_id: int, file_model: File) -> Optional[File]:
        """Add a file to a project."""
        pass

    @abstractmethod
    def get_file(
        self, project_id: int, file_path: str, filename: str
    ) -> Optional[File]:
        """Get a file from a project."""
        pass

    @abstractmethod
    def delete_file(self, file_id: int) -> bool:
        """Delete a file from the database."""
        pass

    @abstractmethod
    def list_files(self, project_id: int) -> List[File]:
        """List all files in a project."""
        pass

    @abstractmethod
    def search_chunks_by_text(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
    ) -> ChunkResults:
        """Search for chunks in a project using text query."""
        pass
