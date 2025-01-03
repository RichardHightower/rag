"""Database file handler for managing projects and files."""

import os
import hashlib
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..embeddings import Embedder
from ..config import DB_URL, CHUNK_SIZE, CHUNK_OVERLAP
from .models import Base, Project, File, Chunk
from .dimension_utils import ensure_vector_dimension
from .chunking import chunk_text


class DBFileHandler:
    """Handler for managing files in the database."""

    def __init__(self, db_url: Optional[str] = None, embedder: Optional[Embedder] = None):
        """Initialize the handler.

        Args:
            db_url: Database URL, defaults to config.DB_URL
            embedder: Embedder instance for generating embeddings
        """
        self.engine = create_engine(db_url or DB_URL)
        self.embedder = embedder
        self.Session = sessionmaker(bind=self.engine)
        
        # Make models accessible
        self.Project = Project
        self.File = File
        self.Chunk = Chunk
        
        # Ensure tables exist
        Base.metadata.create_all(self.engine)
        
        # Ensure vector dimension matches embedder if provided
        if embedder:
            ensure_vector_dimension(self.engine, embedder.get_dimension())
    
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

    def get_or_create_project(self, name: str, description: str = None) -> Project:
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
                    updated_at=project.updated_at
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
                updated_at=project.updated_at
            )
            return project_copy

    def create_project(self, name: str, description: str = None) -> Project:
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
                updated_at=project.updated_at
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
                    updated_at=project.updated_at
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

    def add_file(self, project_id: int, file_path: str, chunk_size: Optional[int] = None, 
                 overlap: Optional[int] = None) -> Optional[File]:
        """Add a file to a project.

        Args:
            project_id: ID of the project
            file_path: Path to the file
            chunk_size: Number of lines per chunk, defaults to config.CHUNK_SIZE
            overlap: Number of lines to overlap between chunks, defaults to config.CHUNK_OVERLAP

        Returns:
            File: Created file object if successful, None if project not found
        """
        if not self.embedder:
            raise ValueError("Embedder must be provided to add files")
            
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP
        
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(Project, project_id)
            if not project:
                return None

            # Read file and compute metadata
            with open(file_path, 'r') as f:
                content = f.read()
            
            file_size = os.path.getsize(file_path)
            crc = hashlib.md5(content.encode()).hexdigest()
            filename = os.path.basename(file_path)

            # Create file record
            file = File(
                project_id=project_id,
                filename=filename,
                file_path=file_path,
                crc=crc,
                file_size=file_size
            )
            session.add(file)
            session.flush()  # Get file.id

            # Create chunks
            chunks = chunk_text(content, chunk_size, overlap)
            embeddings = self.embedder.embed_texts(chunks)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_obj = Chunk(
                    file_id=file.id,
                    content=chunk,
                    embedding=embedding,
                    chunk_index=idx
                )
                session.add(chunk_obj)
            
            # Get a copy of the file data
            file_data = {
                'id': file.id,
                'project_id': file.project_id,
                'filename': file.filename,
                'file_path': file.file_path,
                'crc': file.crc,
                'file_size': file.file_size,
                'created_at': file.created_at
            }
            
            # Commit to ensure the data is saved
            session.commit()
            
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
