"""Database file handler."""

import hashlib
import logging
import os
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from ..chunking import split_text_into_chunks
from ..embeddings.base_embedder import BaseEmbedder
from .models import Base, Chunk, File, Project

logger = logging.getLogger(__name__)


class DBFileHandler:
    """Database file handler."""

    def __init__(self, db_url: str, embedder: BaseEmbedder):
        """Initialize database file handler."""
        self.engine = create_engine(
            db_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
        )
        self.Session = sessionmaker(bind=self.engine)
        self.embedder = embedder
        self.Project = Project
        self.File = File
        self.Chunk = Chunk

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error in session: {str(e)}")
            raise
        finally:
            session.close()

    def calculate_crc(self, file_content: str) -> int:
        """Calculate CRC32 checksum of file content."""
        return zlib.crc32(file_content.encode("utf-8"))

    def get_or_create_project(
        self, name: str, description: Optional[str] = None
    ) -> Project:
        """Get existing project or create a new one."""
        with self.session_scope() as session:
            project = session.query(self.Project).filter(self.Project.name == name).first()
            if not project:
                project = self.Project(
                    name=name,
                    description=description,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(project)
                session.flush()

            # Create a detached copy of the project
            detached_project = self.Project(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
            return detached_project

    def process_file(
        self,
        project_id: int,
        file_path: str,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Tuple[Optional[File], bool]:
        """
        Process a file and store its chunks with embeddings.
        Returns tuple of (file, is_updated) where is_updated indicates if an existing file was updated.
        If project_id doesn't exist, returns (None, False).
        """
        filename = os.path.basename(file_path)
        file_type = os.path.splitext(filename)[1]
        current_crc = self.calculate_crc(content)
        file_size = len(content.encode("utf-8"))

        with self.session_scope() as session:
            # First check if project exists
            project = session.get(self.Project, project_id)
            if not project:
                logger.warning(f"Project {project_id} not found")
                return None, False

            # Check if file exists
            existing_file = (
                session.query(self.File)
                .filter(
                    self.File.filename == filename,
                    self.File.file_path == file_path,
                    self.File.project_id == project_id,
                )
                .first()
            )

            is_updated = False
            if existing_file:
                if existing_file.crc != current_crc:
                    logger.info(f"File {file_path} has changed, updating chunks...")
                    # Delete existing chunks
                    session.query(self.Chunk).filter(
                        self.Chunk.file_id == existing_file.id
                    ).delete()
                    # Update file
                    existing_file.crc = current_crc
                    existing_file.file_size = file_size  # Update file size
                    existing_file.updated_at = datetime.now(timezone.utc)
                    file = existing_file
                    is_updated = True
                else:
                    logger.info(f"File {file_path} unchanged, skipping...")
                    # Create a detached copy of the file
                    detached_file = self.File(
                        id=existing_file.id,
                        filename=existing_file.filename,
                        file_path=existing_file.file_path,
                        file_type=existing_file.file_type,
                        crc=existing_file.crc,
                        file_size=existing_file.file_size,  # Include file size
                        project_id=existing_file.project_id,
                        created_at=existing_file.created_at,
                        updated_at=existing_file.updated_at,
                    )
                    return detached_file, False
            else:
                logger.info(f"Creating new file entry for {file_path}")
                file = self.File(
                    filename=filename,
                    file_path=file_path,
                    file_type=file_type,
                    crc=current_crc,
                    file_size=file_size,  # Set file size for new files
                    project_id=project_id,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(file)
                session.flush()

            # Process chunks
            chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
            embeddings = self.embedder.embed_documents(
                [chunk.content for chunk in chunks]
            )

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db_chunk = self.Chunk(
                    content=chunk.content,
                    start_char=chunk.start,
                    end_char=chunk.end,
                    embedding=embedding,
                    file_id=file.id,
                    chunk_index=idx,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(db_chunk)

            session.flush()

            # Create a detached copy of the file
            detached_file = self.File(
                id=file.id,
                filename=file.filename,
                file_path=file.file_path,
                file_type=file.file_type,
                crc=file.crc,
                file_size=file.file_size,  # Include file size
                project_id=file.project_id,
                created_at=file.created_at,
                updated_at=file.updated_at,
            )
            return detached_file, is_updated
