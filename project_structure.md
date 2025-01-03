# .

## 

### setup.py

```python
"""Setup file for the rag package."""

from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.58.1",
        "sqlalchemy>=2.0.36",
        "psycopg2-binary>=2.9.10",
        "pgvector>=0.3.6",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
        ]
    },
    python_requires=">=3.10",
)

```
## tests

### tests/conftest.py

```python
"""Pytest configuration file."""

import os
from pathlib import Path
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from rag.db.models import Base
from rag.embeddings import MockEmbedder
from rag.config import DB_URL

# Load environment variables at the start of testing
load_dotenv()

def get_db_url(dbname: str) -> str:
    """Get database URL with specific database name."""
    return f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{dbname}"


def create_database(db_url: str, template_db_url: str):
    """Create a database if it doesn't exist."""
    engine = create_engine(template_db_url)
    try:
        with engine.connect() as conn:
            # Disconnect all users from the database we will drop
            conn.execute(text(
                """
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = 'vectordb_test'
                """
            ))
            conn.execute(text("commit"))
            
            # Drop and recreate the database
            conn.execute(text("DROP DATABASE IF EXISTS vectordb_test"))
            conn.execute(text("commit"))
            conn.execute(text("CREATE DATABASE vectordb_test"))
            conn.execute(text("commit"))

            # Enable pgvector extension
            test_engine = create_engine(db_url)
            with test_engine.connect() as test_conn:
                test_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                test_conn.execute(text("commit"))
    except OperationalError as e:
        pytest.fail(f"Failed to connect to database. Please check your environment variables and ensure the database is running: {e}")


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=4)  # Small dimension for testing


@pytest.fixture
def test_db():
    """Create a test database."""
    # Get URLs for test and template databases
    test_db_url = get_db_url("vectordb_test")
    template_db_url = get_db_url("postgres")
    
    try:
        # Create the test database
        create_database(test_db_url, template_db_url)
        
        # Create the engine and tables
        engine = create_engine(test_db_url)
        Base.metadata.create_all(engine)
        
        yield engine
        
        # Cleanup
        Base.metadata.drop_all(engine)
    except Exception as e:
        pytest.fail(f"Test database setup failed. Error: {e}")

```
## tests/embeddings

### tests/embeddings/test_embedders.py

```python
"""Test embedders."""

import pytest
from rag.embeddings import MockEmbedder


def test_mock_embedder():
    """Test mock embedder."""
    embedder = MockEmbedder(dimension=4)
    texts = ["Hello, world!", "Another test"]
    
    embeddings = embedder.embed_texts(texts)
    
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 4
    assert all(-1 <= x <= 1 for x in embeddings[0])

```
## tests/db

### tests/db/test_db_file_handler.py

```python
"""Test database file handler."""

import pytest
from rag.db.db_file_handler import DBFileHandler


def test_create_project(test_db):
    """Test creating a project."""
    handler = DBFileHandler(str(test_db.url))
    
    project = handler.create_project("Test Project", "Test Description")
    
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.description == "Test Description"


def test_delete_project(test_db):
    """Test deleting a project."""
    handler = DBFileHandler(str(test_db.url))
    
    project = handler.create_project("Test Project")
    project_id = project.id
    
    handler.delete_project(project_id)
    
    # Verify project is deleted
    session = handler.Session()
    deleted_project = session.query(handler.Project).get(project_id)
    session.close()
    
    assert deleted_project is None


def test_add_file_requires_embedder(test_db):
    """Test that adding a file requires an embedder."""
    handler = DBFileHandler(str(test_db.url))
    project = handler.create_project("Test Project")
    
    with pytest.raises(ValueError, match="Embedder must be provided to add files"):
        handler.add_file(project.id, "test.txt")

```
## db

### db/init.sql

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

```
## db/sql/ddl

### db/sql/ddl/01_init_tables.sql

```sql
-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Projects table to store different RAG projects
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Files table to store source documents
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    crc VARCHAR(32) NOT NULL,
    file_size BIGINT NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_ingested TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, file_path)
);

-- Chunks table to store embeddings and text chunks
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI's default embedding size
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(file_id, chunk_index)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Create a function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update timestamps
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at
    BEFORE UPDATE ON files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

```
## src/rag

### src/rag/config.py

```python
"""Configuration module for RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
DB_NAME = os.getenv("POSTGRES_DB", "vectordb")

# Construct database URL
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# Vector dimensions
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # OpenAI's default

# File processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

```
### src/rag/__init__.py

```python
"""RAG (Retrieval Augmented Generation) package."""

```
## src/rag/embeddings

### src/rag/embeddings/openai_embedder.py

```python
"""OpenAI embedder implementation."""

from typing import List
import openai
from .base import Embedder
from ..config import OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_DIM


class OpenAIEmbedder(Embedder):
    """OpenAI embedder implementation."""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        dimension: int = EMBEDDING_DIM,
        api_key: str = OPENAI_API_KEY,
        batch_size: int = 16,
    ):
        """Initialize OpenAI embedder.

        Args:
            model_name: Name of the OpenAI model to use
            dimension: Dimension of the embeddings
            api_key: OpenAI API key
            batch_size: Number of texts to embed in one batch
        """
        super().__init__(model_name, dimension)
        if not api_key:
            raise ValueError("OpenAI API key must be provided")
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = batch_size

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

```
### src/rag/embeddings/__init__.py

```python
"""Embeddings package for handling text embeddings from various sources."""

from .base import Embedder
from .openai_embedder import OpenAIEmbedder
from .mock_embedder import MockEmbedder

__all__ = ['Embedder', 'OpenAIEmbedder', 'MockEmbedder']

```
### src/rag/embeddings/mock_embedder.py

```python
"""Mock embedder for testing."""

import random
from typing import List
from .base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder that returns random vectors."""

    def __init__(self, model_name: str = "mock", dimension: int = 1536):
        """Initialize mock embedder.

        Args:
            model_name: Name of the mock model
            dimension: Dimension of the embeddings
        """
        super().__init__(model_name, dimension)

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of random embeddings, one per text
        """
        return [
            [random.uniform(-1, 1) for _ in range(self.dimension)]
            for _ in texts
        ]

```
### src/rag/embeddings/base.py

```python
"""Base embedder interface."""

from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """Base class for all embedders."""

    def __init__(self, model_name: str, dimension: int):
        """Initialize embedder with model name and dimension.

        Args:
            model_name: Name of the model to use
            dimension: Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        pass

```
## src/rag/db

### src/rag/db/chunking.py

```python
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
    lines = text.splitlines()
    chunks = []
    start = 0

    while start < len(lines):
        # Calculate end of current chunk
        end = min(start + chunk_size, len(lines))
        
        # Join lines for this chunk
        chunk = '\n'.join(lines[start:end])
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap

    return chunks

```
### src/rag/db/models.py

```python
"""SQLAlchemy models for RAG."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Project(Base):
    """Project model."""

    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    files = relationship('File', back_populates='project', cascade='all, delete-orphan')


class File(Base):
    """File model."""

    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    crc = Column(String(32), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_ingested = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    project = relationship('Project', back_populates='files')
    chunks = relationship('Chunk', back_populates='file', cascade='all, delete-orphan')


class Chunk(Base):
    """Chunk model with vector embedding."""

    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Default to OpenAI's dimension
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    file = relationship('File', back_populates='chunks')

```
### src/rag/db/__init__.py

```python
"""Database package for RAG."""

from .models import Project, File, Chunk
from .dimension_utils import ensure_vector_dimension
from .db_file_handler import DBFileHandler

__all__ = ['Project', 'File', 'Chunk', 'ensure_vector_dimension', 'DBFileHandler']

```
### src/rag/db/dimension_utils.py

```python
"""Utilities for managing vector dimensions in the database."""

from sqlalchemy import text


def ensure_vector_dimension(engine, desired_dim: int):
    """Ensure the chunks.embedding column has the correct dimension.

    Args:
        engine: SQLAlchemy engine
        desired_dim: Desired embedding dimension

    Note:
        This will alter the table if the dimension doesn't match.
        Be careful with existing data when changing dimensions.
    """
    with engine.connect() as conn:
        # Check current dimension
        result = conn.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """))
        current_dim = result.scalar()

        if current_dim != desired_dim:
            conn.execute(text(f"""
                ALTER TABLE chunks
                ALTER COLUMN embedding TYPE vector({desired_dim})
                USING embedding::vector({desired_dim});
            """))
            conn.commit()

```
### src/rag/db/db_file_handler.py

```python
"""Database file handler for managing projects and files."""

import os
import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
        
        # Ensure tables exist
        Base.metadata.create_all(self.engine)
        
        # Ensure vector dimension matches embedder if provided
        if embedder:
            ensure_vector_dimension(self.engine, embedder.get_dimension())

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Project: Created project
        """
        session = self.Session()
        try:
            project = Project(name=name, description=description)
            session.add(project)
            session.commit()
            return project
        finally:
            session.close()

    def delete_project(self, project_id: int):
        """Delete a project and all its files.

        Args:
            project_id: ID of the project to delete
        """
        session = self.Session()
        try:
            project = session.query(Project).get(project_id)
            if project:
                session.delete(project)
                session.commit()
        finally:
            session.close()

    def add_file(self, project_id: int, file_path: str, chunk_size: Optional[int] = None, 
                 overlap: Optional[int] = None):
        """Add a file to a project.

        Args:
            project_id: ID of the project
            file_path: Path to the file
            chunk_size: Number of lines per chunk, defaults to config.CHUNK_SIZE
            overlap: Number of lines to overlap between chunks, defaults to config.CHUNK_OVERLAP
        """
        if not self.embedder:
            raise ValueError("Embedder must be provided to add files")
            
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP
        
        session = self.Session()
        try:
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

            session.commit()
        finally:
            session.close()

    def remove_file(self, project_id: int, file_id: int):
        """Remove a file from a project.

        Args:
            project_id: ID of the project
            file_id: ID of the file to remove
        """
        session = self.Session()
        try:
            file = session.query(File).filter_by(
                id=file_id, 
                project_id=project_id
            ).first()
            if file:
                session.delete(file)
                session.commit()
        finally:
            session.close()

```
