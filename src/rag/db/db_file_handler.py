"""Database file handler for managing projects and files."""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rag.model import Chunk as ChunkModel
from rag.model import File as FileModel

from ..chunking import LineChunker
from ..config import DB_URL
from ..embeddings import Embedder, OpenAIEmbedder
from .dimension_utils import ensure_vector_dimension
from .models import Base, Chunk, File, Project

logger = logging.getLogger(__name__)


class DBFileHandler:
    """Handler for managing files in the database."""

    def __init__(self, db_url: str = DB_URL, embedder: Optional[Embedder] = None):
        """Initialize the handler.

        Args:
            db_url: Database URL, defaults to config.DB_URL
            embedder: Embedder instance for generating embeddings

        Raises:
            ValueError: If db_url is None
        """
        if db_url is None:
            raise ValueError("Database URL must be provided")

        self.engine = create_engine(db_url)
        self.embedder = embedder or OpenAIEmbedder()
        self.Session = sessionmaker(bind=self.engine)
        self.chunker = LineChunker()

        # Make models accessible
        self.Project = Project
        self.File = File
        self.Chunk = Chunk

        # Ensure tables exist
        Base.metadata.create_all(self.engine)

        # Ensure vector dimension matches embedder if provided
        if self.embedder:
            ensure_vector_dimension(self.engine, self.embedder.get_dimension())

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_or_create_project(
        self, name: str, description: Optional[str] = None
    ) -> Project:
        """Get an existing project by name or create a new one.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Project instance
        """
        with self.session_scope() as session:
            project = session.query(Project).filter(Project.name == name).first()
            if project:
                if description and description != project.description:
                    project.description = description
                    project.updated_at = datetime.now(timezone.utc)
                session.flush()
                # Create a detached copy with all attributes loaded
                project_copy = Project(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
                return project_copy

            project = Project(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
            return project_copy

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Project instance

        Raises:
            ValueError: If project with given name already exists
        """
        with self.session_scope() as session:
            existing = session.query(Project).filter(Project.name == name).first()
            if existing:
                raise ValueError(f"Project with name '{name}' already exists")

            project = Project(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
            return project_copy

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a project by ID.

        Args:
            project_id: ID of the project

        Returns:
            Project if found, None otherwise
        """
        with self.session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                # Get a copy of the data
                return Project(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
            return None

    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all its files.

        Args:
            project_id: ID of the project to delete

        Returns:
            bool: True if project was deleted, False if not found
        """
        with self.session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                session.delete(project)
                return True
            return False

    def add_file(self, project_id: int, file_model: FileModel) -> Optional[File]:
        """Add a file to a project with version checking.

        Args:
            project_id: ID of the project
            file_model: File model containing file information

        Returns:
            File: Created or existing file record
            None: If project doesn't exist
        """
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(Project, project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None

            # Check if file already exists
            existing_file = self.get_file(project_id, file_model.path, file_model.name)

            if existing_file:
                if existing_file.crc == file_model.crc:
                    logger.info(
                        f"File {file_model.name} already exists with same CRC {file_model.crc}"
                    )
                    return existing_file
                else:
                    logger.info(
                        f"File {file_model.name} exists but CRC differs "
                        f"(old: {existing_file.crc}, new: {file_model.crc})"
                    )
                    logger.info("Deleting old version and creating new version")
                    self.delete_file(existing_file.id)

            # Create new file record
            file = File(
                project_id=project_id,
                filename=file_model.name,
                file_path=file_model.path,
                crc=file_model.crc,
                file_size=file_model.size,
            )
            session.add(file)
            session.flush()  # Get file.id

            # Create chunks
            chunks: List[ChunkModel] = self.chunker.chunk_text(file_model)
            embeddings = self.embedder.embed_texts(chunks)

            for chunk, embedding in zip(chunks, embeddings):
                chunk_obj = Chunk(
                    file_id=file.id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                )
                session.add(chunk_obj)

            # Get a copy of the file data
            file_data = {
                "id": file.id,
                "project_id": file.project_id,
                "filename": file.filename,
                "file_path": file.file_path,
                "crc": file.crc,
                "file_size": file.file_size,
                "created_at": file.created_at,
            }

            # Commit to ensure the data is saved
            session.commit()
            logger.info(
                f"Successfully added file {file_model.name} to project {project_id}"
            )

            # Return a new instance with the data
            return File(**file_data)

    def remove_file(self, project_id: int, file_id: int) -> bool:
        """Remove a file from a project.

        Args:
            project_id: ID of the project
            file_id: ID of the file to remove

        Returns:
            bool: True if file was removed, False if not found
        """
        with self.session_scope() as session:
            file = session.get(File, file_id)
            if file and file.project_id == project_id:
                session.delete(file)
                return True
            return False

    def delete_file(self, file_id: int) -> bool:
        """Delete a file and all its associated chunks from the database.

        Args:
            file_id: ID of the file to delete

        Returns:
            bool: True if file was deleted, False if not found
        """
        with self.session_scope() as session:
            file = session.get(self.File, file_id)
            if file is None:
                return False

            session.delete(file)
            session.flush()
            return True

    def get_file(
        self, project_id: int, file_path: str, filename: str
    ) -> Optional[File]:
        """Look up a file by project ID, path and name.

        Args:
            project_id: ID of the project containing the file
            file_path: Full path of the file
            filename: Name of the file

        Returns:
            File if found, None otherwise
        """
        with self.session_scope() as session:
            file = (
                session.query(File)
                .filter(File.project_id == project_id)
                .filter(File.file_path == file_path)
                .filter(File.filename == filename)
                .first()
            )

            if file:
                # Return a copy of the file data
                return File(
                    id=file.id,
                    project_id=file.project_id,
                    filename=file.filename,
                    file_path=file.file_path,
                    crc=file.crc,
                    file_size=file.file_size,
                    last_updated=file.last_updated,
                    last_ingested=file.last_ingested,
                    created_at=file.created_at,
                )
            else:
                return None
