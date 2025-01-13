"""Database file handler for managing projects and files."""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Union

import numpy as np
from sqlalchemy import Float, create_engine, func, literal, select
from sqlalchemy.orm import sessionmaker

from rag.model import Chunk, ChunkResult, ChunkResults, File, Project

from ..chunking import LineChunker
from ..chunking.base_chunker import Chunker
from ..config import DB_URL
from ..embeddings import Embedder, OpenAIEmbedder
from .base_file_handler import FileHandler
from .db_model import ChunkDB, DbBase, FileDB, ProjectDB
from .dimension_utils import ensure_vector_dimension

logger = logging.getLogger(__name__)


class DBFileHandler(FileHandler):
    """Handler for managing files in the database."""

    def __init__(
        self,
        db_url: str = DB_URL,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
    ):
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
        self.chunker: Chunker

        if not chunker:
            self.chunker = LineChunker()
        else:
            self.chunker = chunker

        # Make models accessible
        self.Project = ProjectDB
        self.File = FileDB
        self.Chunk = ChunkDB

        # Ensure tables exist
        DbBase.metadata.create_all(self.engine)

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
            project = session.query(ProjectDB).filter(ProjectDB.name == name).first()
            if project:
                if description and description != project.description:
                    project.description = description
                    project.updated_at = datetime.now(timezone.utc)
                session.flush()
                # Create a detached copy with all attributes loaded
                project_copy = Project(
                    name=project.name, description=project.description, id=project.id
                )
                return project_copy

            project = ProjectDB(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                name=project.name, description=project.description, id=project.id
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
            existing = session.query(ProjectDB).filter(ProjectDB.name == name).first()
            if existing:
                raise ValueError(f"Project with name '{name}' already exists")

            project = ProjectDB(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                name=project.name, description=project.description, id=project.id
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
            project = session.get(ProjectDB, project_id)
            if project:
                # Get a copy of the data
                return Project(
                    name=project.name, description=project.description, id=project.id
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
            project = session.get(ProjectDB, project_id)
            if project:
                session.delete(project)
                return True
            return False

    def add_chunk(self, file_id: int, chunk: Chunk) -> Optional[Chunk]:
        """Add a single chunk to the database for a given file.

        Args:
            file_id: ID of the file this chunk belongs to
            chunk: Chunk object to be added

        Returns:
            Chunk: Created chunk object
            None: If file doesn't exist or an error occurs
        """
        with self.session_scope() as session:
            # Verify file exists
            file = session.get(FileDB, file_id)
            if not file:
                logger.error(f"File with id {file_id} not found")
                return None

            try:
                # Generate embedding for the chunk
                embedding = self.embedder.embed_texts([chunk])[0]

                # Create new chunk record
                chunk_db = ChunkDB(
                    file_id=file_id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                    chunk_metadata=chunk.meta_data,
                )
                session.add(chunk_db)
                session.flush()  # Get chunk_db.id

                # Create and return a Chunk object
                return Chunk(
                    target_size=chunk.target_size,
                    content=chunk_db.content,
                    index=chunk_db.chunk_index,
                    meta_data=chunk_db.chunk_metadata,
                )

            except Exception as e:
                logger.error(f"Error adding chunk to file {file_id}: {str(e)}")
                return None

    def add_file(self, project_id: int, file_model: File) -> Optional[File]:
        """Add a file to a project with version checking.

        Args:
            project_id: ID of the project
            file_model: File model containing file information

        Returns:
            FileDB: Created or existing file record
            None: If project doesn't exist
        """
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(ProjectDB, project_id)
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
                    if existing_file.id is not None:
                        self.delete_file(existing_file.id)

            # Create new file record
            file = FileDB(
                project_id=project_id,
                filename=file_model.name,
                file_path=file_model.path,
                crc=file_model.crc,
                file_size=file_model.size,
            )
            session.add(file)
            session.flush()  # Get file.id

            # Create chunks
            chunks: List[Chunk] = self.chunker.chunk_text(file_model)
            embeddings = self.embedder.embed_texts(chunks)

            for chunk, embedding in zip(chunks, embeddings):
                chunk_obj = ChunkDB(
                    file_id=file.id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                )
                session.add(chunk_obj)

            # Get a copy of the file data
            file_data = File(
                id=file.id,
                name=file.filename,
                path=file.file_path,
                file_size=file.file_size,
                crc=file.crc,
            )

            # Commit to ensure the data is saved
            session.commit()
            logger.info(
                f"Successfully added file {file_model.name} to project {project_id}"
            )

            # Return a new instance with the data
            return file_data

    def remove_file(self, project_id: int, file_id: int) -> bool:
        """Remove a file from a project.

        Args:
            project_id: ID of the project
            file_id: ID of the file to remove

        Returns:
            bool: True if file was removed, False if not found
        """
        with self.session_scope() as session:
            file = session.get(FileDB, file_id)
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
                session.query(FileDB)
                .filter(FileDB.project_id == project_id)
                .filter(FileDB.file_path == file_path)
                .filter(FileDB.filename == filename)
                .first()
            )

            if file:
                # Return a copy of the file data
                return File(
                    id=file.id,
                    name=file.filename,
                    path=file.file_path,
                    file_size=file.file_size,
                    crc=file.crc,
                )
            else:
                return None

    def get_projects(self, limit: int = -1, offset: int = -1) -> List[ProjectDB]:
        """Get a list of all projects.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip

        Returns:
            List[ProjectDB]: List of projects ordered by creation date (newest first)
        """
        with self.session_scope() as session:
            query = session.query(ProjectDB).order_by(ProjectDB.created_at.desc())

            if limit != -1:
                query = query.limit(limit)
            if offset != -1:
                query = query.offset(offset)

            projects = query.all()

            # Create detached copies of the projects
            return [
                ProjectDB(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
                for project in projects
            ]

    def list_files(self, project_id: int) -> List[File]:
        """List all files in a project.

        Args:
            project_id: ID of the project

        Returns:
            List[FileModel]: List of file models, empty list if project doesn't exist
        """
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(self.Project, project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return []

            # Query all files for the project
            db_files = (
                session.query(self.File)
                .filter(self.File.project_id == project_id)
                .all()
            )

            # Convert DB models to FileModel instances
            files = []
            for db_file in db_files:
                # Get all chunks for this file, ordered by chunk_index
                chunks = (
                    session.query(self.Chunk)
                    .filter(self.Chunk.file_id == db_file.id)
                    .order_by(self.Chunk.chunk_index)
                    .all()
                )

                # Reconstruct original content from chunks
                content = "\n".join(chunk.content for chunk in chunks)

                # Create FileModel instance
                file_model = File(
                    id=db_file.id,
                    name=db_file.filename,
                    path=db_file.file_path,
                    crc=db_file.crc,
                    content=content,
                    meta_data={
                        "type": (
                            db_file.filename.split(".")[-1]
                            if "." in db_file.filename
                            else ""
                        )
                    },
                )
                files.append(file_model)

            return files

    def search_chunks_by_text(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
    ) -> ChunkResults:
        """Search for chunks in a project using text query with pagination."""
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        if page_size < 1:
            raise ValueError("Page size must be greater than 1")

        # Get embedding for query text
        query_embedding = self.embedder.embed_texts(
            [Chunk(target_size=1, content=query_text, index=0)]
        )[0]

        return self.search_chunks_by_embedding(
            project_id, query_embedding, page, page_size, similarity_threshold
        )

    def search_chunks_by_embedding(
        self,
        project_id: int,
        embedding: Union[np.ndarray, Sequence[float]],
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
    ) -> ChunkResults:
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        if page_size < 1:
            raise ValueError("Page size must be greater than 1")

        # Ensure `embedding` is a 1D float32 (big-endian) array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=">f4")
        elif embedding.dtype != ">f4":
            embedding = embedding.astype(">f4")
        embedding = embedding.ravel()

        with self.session_scope() as session:
            # distance_expr is chunks.embedding <=> your_query_embedding
            distance_expr = self.Chunk.embedding.op("<=>")(embedding)

            # Mark 1.0 as a float literal so that it doesn't become a vector
            similarity_expr = (literal(1.0, type_=Float) - distance_expr).label(
                "similarity"
            )

            # Also mark threshold as a float literal if needed
            threshold_expr = literal(similarity_threshold, type_=Float)

            # Build query
            base_query = (
                select(self.Chunk, similarity_expr)
                .join(self.File)
                .where(self.File.project_id == project_id)
                .where(similarity_expr >= threshold_expr)  # numeric comparison
            )

            # Count how many total rows match
            count_query = select(func.count()).select_from(base_query.subquery())
            total_count = session.execute(count_query).scalar() or 0

            # Pagination
            offset = (page - 1) * page_size
            results = session.execute(
                base_query.order_by(similarity_expr.desc())
                .offset(offset)
                .limit(page_size)
            ).all()

            # Convert to your Pydantic "ChunkResults"
            chunk_results = []
            for chunk_row, similarity in results:
                chunk_results.append(
                    ChunkResult(
                        score=float(similarity),
                        chunk=Chunk(
                            target_size=1,
                            content=chunk_row.content,
                            index=chunk_row.chunk_index,
                            meta_data=chunk_row.chunk_metadata,
                        ),
                    )
                )

            return ChunkResults(
                results=chunk_results,
                total_count=total_count,
                page=page,
                page_size=page_size,
            )
