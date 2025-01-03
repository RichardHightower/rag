# .

## 

# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that processes text files, generates embeddings, and stores them in a Postgres database with pgvector for efficient similarity search.

For detailed information, see:
- [Design Document](DESIGN.md) - System architecture and requirements
- [Developer Guide](developer_guide.md) - Detailed setup and development instructions

## Features

- File ingestion (source code, Markdown, plain text)
- Text chunking with configurable overlap
- Vector embeddings via OpenAI or Hugging Face
- Postgres + pgvector for vector storage and search
- Project-based organization of documents

## Quick Start

1. Set up the environment:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings:
# - Database credentials
# - OpenAI API key (if using OpenAI embeddings)
```

3. Start the database:
```bash
task db:up
```

4. Run the example:
```bash
task demo:example
```

## Development

```bash
# Run tests
task test:integration

```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


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
            "pytest-cov===6.0.0"
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
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from rag.embeddings import MockEmbedder


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=4)  # Small dimension for testing


@pytest.fixture(scope="session")
def test_db():
    """Create a test database engine."""
    # Get database credentials from environment variables
    password = os.environ.get("POSTGRES_PASSWORD", "postgres").strip()
    username = os.environ.get("POSTGRES_USER", "postgres").strip()
    host = os.environ.get("POSTGRES_HOST", "localhost").strip()
    port = int(os.environ.get("POSTGRES_PORT", "5433").strip())
    database = "vectordb_test"
    db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"


    print(f"POSTGRES_USER={username}")
    print(f"POSTGRES_PASSWORD={password}")
    print(f"POSTGRES_HOST={host}")
    print(f"POSTGRES_PORT={port}")
    print(f"Database URL={db_url}")

    # Create the engine with minimal pooling
    engine = create_engine(
        db_url,
        poolclass=NullPool,
        connect_args={"connect_timeout": 5}
    )
    
    # Drop all tables first to ensure clean state
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))  # Add pgvector extension
        conn.commit()
    
    # Create all tables
    from rag.db.models import Base
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Drop all tables after tests
    Base.metadata.drop_all(engine)

```
## tests/unit

### tests/unit/test_chunking.py

```python
"""Tests for text chunking functionality."""

import pytest
from rag.chunking import split_text_into_chunks


def test_empty_text():
    """Test chunking empty text."""
    chunks = split_text_into_chunks("")
    assert len(chunks) == 0


def test_text_smaller_than_chunk_size():
    """Test chunking text smaller than chunk size."""
    text = "Small text"
    chunks = split_text_into_chunks(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)


def test_text_larger_than_chunk_size():
    """Test chunking text larger than chunk size."""
    text = "This is a longer text that should be split into multiple chunks based on the specified chunk size."
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    # Check that chunks cover all text
    full_text = ""
    for chunk in chunks:
        full_text += chunk.text + " "
    assert text in full_text.strip()


def test_chunk_overlap():
    """Test that chunks properly overlap."""
    text = "This is a test of the chunking overlap functionality"
    chunk_size = 20
    overlap = 10
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    for i in range(len(chunks) - 1):
        # Get the overlapping text
        chunk1_end = chunks[i].text[-overlap:]
        chunk2_start = chunks[i + 1].text[:overlap]
        # There should be some common text between consecutive chunks
        assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0


def test_no_tiny_final_chunk():
    """Test that very small remaining text is merged with previous chunk."""
    text = "This is a test text that should not create a tiny final chunk"
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    # The last chunk should not be very small
    assert len(chunks[-1].text) > overlap

```
## tests/embeddings

### tests/embeddings/test_openai_embedder.py

```python
"""Test OpenAI embedder."""

import os
from unittest.mock import patch, Mock
import pytest
from rag.embeddings.openai_embedder import OpenAIEmbedder
from rag.config import EMBEDDING_DIM, OPENAI_MODEL


def test_init_with_api_key():
    """Test initialization with API key."""
    embedder = OpenAIEmbedder(api_key="test_key")
    assert embedder.model_name == OPENAI_MODEL
    assert embedder.dimension == EMBEDDING_DIM
    assert embedder.batch_size == 16


def test_init_without_api_key():
    """Test initialization without API key."""
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAIEmbedder(api_key="")


def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    embedder = OpenAIEmbedder(
        model_name="custom-model",
        dimension=128,
        api_key="test_key",
        batch_size=32
    )
    assert embedder.model_name == "custom-model"
    assert embedder.dimension == 128
    assert embedder.batch_size == 32


def test_get_dimension():
    """Test get_dimension method."""
    embedder = OpenAIEmbedder(api_key="test_key", dimension=256)
    assert embedder.get_dimension() == 256


@patch('openai.OpenAI')
def test_embed_texts_single_batch(mock_openai):
    """Test embedding texts that fit in a single batch."""
    # Mock response from OpenAI
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_client = Mock()
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    embedder = OpenAIEmbedder(api_key="test_key", dimension=3, batch_size=5)
    texts = ["Hello", "World"]
    embeddings = embedder.embed_texts(texts)

    # Verify the results
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify OpenAI was called correctly
    mock_client.embeddings.create.assert_called_once_with(
        model=OPENAI_MODEL,
        input=texts
    )


@patch('openai.OpenAI')
def test_embed_texts_multiple_batches(mock_openai):
    """Test embedding texts that require multiple batches."""
    # Mock responses for two batches
    mock_response1 = Mock()
    mock_response1.data = [
        Mock(embedding=[0.1, 0.2]),
        Mock(embedding=[0.3, 0.4])
    ]
    mock_response2 = Mock()
    mock_response2.data = [
        Mock(embedding=[0.5, 0.6])
    ]

    mock_client = Mock()
    mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]
    mock_openai.return_value = mock_client

    embedder = OpenAIEmbedder(api_key="test_key", dimension=2, batch_size=2)
    texts = ["Text1", "Text2", "Text3"]
    embeddings = embedder.embed_texts(texts)

    # Verify the results
    assert len(embeddings) == 3
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]
    assert embeddings[2] == [0.5, 0.6]

    # Verify OpenAI was called correctly for each batch
    assert mock_client.embeddings.create.call_count == 2
    mock_client.embeddings.create.assert_any_call(
        model=OPENAI_MODEL,
        input=["Text1", "Text2"]
    )
    mock_client.embeddings.create.assert_any_call(
        model=OPENAI_MODEL,
        input=["Text3"]
    )

```
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
## tests/integration

### tests/integration/test_db_connection.py

```python
"""Simple script to test database connection."""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
from rag.db.models import Base


def test_psycopg2_connection():
    """Test connection using psycopg2 directly."""
    print("\nTesting psycopg2 connection...")
    try:
        conn = psycopg2.connect(
            dbname="vectordb_test",
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433"))
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        print(f"psycopg2 connection successful: {result}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"psycopg2 connection failed: {str(e)}")


def test_sqlalchemy_connection():
    """Test connection using SQLAlchemy."""
    print("\nTesting SQLAlchemy connection...")
    try:
        # Use the same URL construction as conftest.py
        db_url = URL.create(
            drivername="postgresql",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
            database="vectordb_test"
        )
        print(f"Connection URL: {db_url}")
        engine = create_engine(db_url, echo=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"SQLAlchemy connection successful: {result}")
    except Exception as e:
        print(f"SQLAlchemy connection failed: {str(e)}")


def test_create_tables():
    """Test creating tables."""
    print("\nTesting table creation...")
    try:
        db_url = URL.create(
            drivername="postgresql",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
            database="vectordb_test"
        )
        engine = create_engine(db_url, echo=True)
        
        # Drop all tables first
        Base.metadata.drop_all(engine)
        print("Dropped all tables")
        
        # Create all tables
        Base.metadata.create_all(engine)
        print("Created all tables")
        
        # Verify tables exist
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)).fetchall()
            print("Tables created:", [r[0] for r in result])
            
    except Exception as e:
        print(f"Table creation failed: {str(e)}")


if __name__ == "__main__":
    test_psycopg2_connection()
    test_sqlalchemy_connection()
    test_create_tables()

```
### tests/integration/test_ingestion.py

```python
"""Integration tests for file ingestion (using the same connection style as test_db_connection)."""

import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Load environment variables
env_file = Path(project_root) / ".env"
load_dotenv(env_file)

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.engine import URL

from rag.db.models import Project, File, Chunk
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder


def _get_db_url() -> str:
    """
    Build a DB URL string using environment variables,
    matching the style from test_db_connection.py
    """
    password = os.environ.get("POSTGRES_PASSWORD", "postgres").strip()
    username = os.environ.get("POSTGRES_USER", "postgres").strip()
    host = os.environ.get("POSTGRES_HOST", "localhost").strip()
    port = int(os.environ.get("POSTGRES_PORT", "5433").strip())
    database = "vectordb_test"
    db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"


    print(f"test ingestion POSTGRES_USER={username}")
    print(f"test ingestion POSTGRES_PASSWORD={password}")
    print(f"test ingestion POSTGRES_HOST={host}")
    print(f"test ingestion POSTGRES_PORT={port}")
    print(f"test ingestion Database URL={db_url}")
    db_url2 = str(
        URL.create(
            drivername="postgresql",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
    )
    print(f"test ingestion Database URL={db_url2}")
    return db_url


def test_file_ingestion():
    """Test basic file ingestion flow."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create project
    project = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Test file ingestion
    content = "This is a test file.\nIt has multiple lines.\nEach line is different."
    file, is_updated = handler.process_file(project.id, "/test/file.txt", content)

    assert not is_updated
    assert file.filename == "file.txt"
    assert file.file_path == "/test/file.txt"

    # Verify chunks were created
    engine = create_engine(db_url)
    with Session(engine) as session:
        chunks = session.query(Chunk).filter(Chunk.file_id == file.id).all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.embedding is not None


def test_project_uniqueness():
    """Test project name uniqueness."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create project with unique name
    name = f"test_project_{uuid.uuid4()}"
    project1 = handler.get_or_create_project(name)

    # Try to create project with the same name
    project2 = handler.get_or_create_project(name)

    # Should return the same project
    assert project1.id == project2.id
    assert project1.name == project2.name


def test_multiple_projects():
    """Test creating multiple projects."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create multiple projects with unique names
    names = [f"test_project_{uuid.uuid4()}" for _ in range(3)]
    projects = [handler.get_or_create_project(name) for name in names]

    # Verify all projects were created with unique IDs
    project_ids = [p.id for p in projects]
    assert len(project_ids) == len(set(project_ids))
    assert len(project_ids) == len(names)


def test_file_deduplication():
    """Test file deduplication logic."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())
    project = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Initial file ingestion
    content1 = "Initial content"
    file1, is_updated = handler.process_file(project.id, "/test/file.txt", content1)
    assert not is_updated

    # Same content - should not update
    file2, is_updated = handler.process_file(project.id, "/test/file.txt", content1)
    assert not is_updated
    assert file1.id == file2.id
    assert file1.crc == file2.crc

    # Modified content - should update
    content2 = "Modified content"
    file3, is_updated = handler.process_file(project.id, "/test/file.txt", content2)
    assert is_updated
    assert file1.id == file3.id
    assert file1.crc != file3.crc

    # Verify chunk update
    engine = create_engine(db_url)
    with Session(engine) as session:
        chunks = session.query(Chunk).filter(Chunk.file_id == file3.id).all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert "Modified" in chunk.content


def test_file_path_uniqueness():
    """Test file path uniqueness constraints."""
    db_url = _get_db_url()
    handler = DBFileHandler(db_url, MockEmbedder())

    # Create two projects
    project1 = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")
    project2 = handler.get_or_create_project(f"test_project_{uuid.uuid4()}")

    # Same file path in different projects - should work
    content = "Test content"
    file1, _ = handler.process_file(project1.id, "/test/file.txt", content)
    file2, _ = handler.process_file(project2.id, "/test/file.txt", content)

    assert file1.id != file2.id

    # Verify both files exist
    engine = create_engine(db_url)
    with Session(engine) as session:
        files = session.query(File).filter(File.file_path == "/test/file.txt").all()
        assert len(files) == 2

```
## tests/db

### tests/db/test_db_file_handler.py

```python
"""Test database file handler."""

import os
import tempfile
from pathlib import Path

import pytest
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder
from rag.config import get_db_url, TEST_DB_NAME


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('Line 1\nLine 2\nLine 3\n')
        return Path(f.name)


def test_create_project(test_db):
    """Test creating a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    
    project = handler.create_project("Test Project", "Test Description")
    
    # Test project was created with correct attributes
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.description == "Test Description"
    
    # Verify project exists in database
    with handler.session_scope() as session:
        db_project = session.get(handler.Project, project.id)
        assert db_project is not None
        assert db_project.name == "Test Project"
        assert db_project.description == "Test Description"


def test_get_project(test_db):
    """Test retrieving a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    
    # Create a project
    created = handler.create_project("Test Project", "Test Description")
    
    # Get the project
    project = handler.get_project(created.id)
    assert project is not None
    assert project.id == created.id
    assert project.name == created.name
    assert project.description == created.description
    
    # Try to get a non-existent project
    assert handler.get_project(999) is None


def test_delete_project(test_db):
    """Test deleting a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    
    # Create project
    project = handler.create_project("Test Project")
    assert project.id is not None
    
    # Verify project exists
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is not None
    
    # Delete project
    assert handler.delete_project(project.id) is True
    
    # Verify project is deleted
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is None


def test_delete_nonexistent_project(test_db):
    """Test deleting a project that doesn't exist."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME))
    assert handler.delete_project(999) is False


def test_add_file(test_db, embedder, sample_text_file):
    """Test adding a file to a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project
    project = handler.create_project("Test Project")
    
    # Add file
    file = handler.add_file(project.id, str(sample_text_file))
    
    # Verify file was added
    assert file is not None
    assert file.project_id == project.id
    assert file.filename == sample_text_file.name
    assert file.file_path == str(sample_text_file)
    assert file.created_at is not None
    
    # Verify chunks were created
    with handler.session_scope() as session:
        db_file = session.get(handler.File, file.id)
        assert db_file is not None
        assert len(db_file.chunks) > 0
        
        # Verify chunk content and embeddings
        for chunk in db_file.chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None


def test_add_file_to_nonexistent_project(test_db, embedder, sample_text_file):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Try to add file to non-existent project
    file = handler.add_file(999, str(sample_text_file))
    assert file is None


def test_remove_file(test_db, embedder, sample_text_file):
    """Test removing a file from a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project and add file
    project = handler.create_project("Test Project")
    file = handler.add_file(project.id, str(sample_text_file))
    
    # Remove file
    assert handler.remove_file(project.id, file.id) is True
    
    # Verify file is removed
    with handler.session_scope() as session:
        assert session.get(handler.File, file.id) is None


def test_remove_nonexistent_file(test_db, embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create project
    project = handler.create_project("Test Project")
    
    # Try to remove non-existent file
    assert handler.remove_file(project.id, 999) is False


def test_remove_file_wrong_project(test_db, embedder, sample_text_file):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Create two projects
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")
    
    # Add file to project1
    file = handler.add_file(project1.id, str(sample_text_file))
    
    # Try to remove file from project2
    assert handler.remove_file(project2.id, file.id) is False


def test_embedder_dimension_handling(test_db, embedder):
    """Test embedder dimension handling."""
    # First handler initializes tables
    handler1 = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    
    # Second handler should work with same dimension
    handler2 = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    assert handler2 is not None


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up the temporary file
    for item in Path().glob('*.txt'):
        if item.is_file() and item.suffix == '.txt':
            try:
                item.unlink()
            except OSError:
                pass

```
### tests/db/test_chunking.py

```python
"""Test text chunking utilities."""

import pytest
from rag.db.chunking import chunk_text


def test_basic_chunking():
    """Test basic text chunking with default parameters."""
    # Create test text with 10 lines
    text = '\n'.join([f'Line {i}' for i in range(10)])
    
    # Default chunk_size=500, overlap=50 should return single chunk
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_custom_chunk_size():
    """Test chunking with custom chunk size."""
    # Create test text with 10 lines
    lines = [f'Line {i}' for i in range(10)]
    text = '\n'.join(lines)
    
    # Set chunk_size to 4 lines, overlap to 1
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    
    # Expected chunks with overlap:
    # Chunk 1: lines 0-3
    # Chunk 2: lines 3-6
    # Chunk 3: lines 6-9
    assert len(chunks) == 3
    
    # Verify first chunk
    assert chunks[0] == '\n'.join(['Line 0', 'Line 1', 'Line 2', 'Line 3'])
    
    # Verify middle chunk has overlap
    assert chunks[1] == '\n'.join(['Line 3', 'Line 4', 'Line 5', 'Line 6'])
    
    # Verify last chunk
    assert chunks[2] == '\n'.join(['Line 6', 'Line 7', 'Line 8', 'Line 9'])


def test_empty_text():
    """Test chunking empty text."""
    chunks = chunk_text('')
    assert len(chunks) == 1
    assert chunks[0] == ''


def test_whitespace_text():
    """Test chunking whitespace text."""
    chunks = chunk_text('   \n  \n  ')
    assert len(chunks) == 1
    assert chunks[0] == '   \n  \n  '


def test_single_line():
    """Test chunking single line of text."""
    text = 'Single line'
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_smaller_than_chunk():
    """Test when text is smaller than chunk size."""
    text = '\n'.join([f'Line {i}' for i in range(5)])
    chunks = chunk_text(text, chunk_size=10, overlap=5)  # Make sure overlap < chunk_size
    assert len(chunks) == 1
    assert chunks[0] == text


def test_no_overlap():
    """Test chunking with no overlap."""
    # Create test text with 6 lines
    lines = [f'Line {i}' for i in range(6)]
    text = '\n'.join(lines)
    
    # Set chunk_size to 2 lines, no overlap
    chunks = chunk_text(text, chunk_size=2, overlap=0)
    
    # Expected chunks:
    # Chunk 1: lines 0-1
    # Chunk 2: lines 2-3
    # Chunk 3: lines 4-5
    assert len(chunks) == 3
    assert chunks[0] == '\n'.join(['Line 0', 'Line 1'])
    assert chunks[1] == '\n'.join(['Line 2', 'Line 3'])
    assert chunks[2] == '\n'.join(['Line 4', 'Line 5'])


def test_invalid_chunk_size():
    """Test invalid chunk size."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0)
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=-1)


def test_invalid_overlap():
    """Test invalid overlap size."""
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_text("test", chunk_size=2, overlap=-1)
    
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("test", chunk_size=2, overlap=2)
    
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("test", chunk_size=2, overlap=3)

```
### tests/db/test_dimension_utils.py

```python
"""Test dimension utilities."""

import pytest
from sqlalchemy import text
from rag.db.dimension_utils import ensure_vector_dimension
from rag.config import get_db_url, TEST_DB_NAME


def test_ensure_vector_dimension_same(test_db):
    """Test when current dimension matches desired dimension."""
    engine = test_db
    
    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """))
        conn.commit()
    
    # Call ensure_vector_dimension with same dimension
    ensure_vector_dimension(engine, 3)
    
    # Verify dimension is still 3
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """))
        assert result.scalar() == 3


def test_ensure_vector_dimension_different(test_db):
    """Test when current dimension differs from desired dimension."""
    engine = test_db
    
    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """))
        conn.commit()
    
    # Change to different dimension
    ensure_vector_dimension(engine, 5)
    
    # Verify dimension was changed
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """))
        assert result.scalar() == 5

```
## db

### db/grant_permissions.sql

```sql
-- Grant necessary permissions to postgres user
ALTER USER postgres WITH SUPERUSER;

-- Grant all privileges on test database
\c vectordb_test;
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
ALTER USER postgres WITH SUPERUSER;

```
### db/init.sql

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create test database if it doesn't exist
CREATE DATABASE vectordb_test;

-- Switch to test database and set up extensions
\c vectordb_test;
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions
ALTER USER postgres WITH SUPERUSER;
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uix_project_name UNIQUE (name)
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

# Find the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from project root .env file
load_dotenv(ENV_FILE)

# Database configuration
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
DB_NAME = os.getenv("POSTGRES_DB", "vectordb")

# Database URLs
def get_db_url(dbname: str = None) -> str:
    """Get database URL with optional database name override."""
    db = dbname or DB_NAME
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db}"
    return db_url

DB_URL = get_db_url()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# Vector dimensions
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # OpenAI's default

# File processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Test configuration
TEST_DB_NAME = "vectordb_test"
TEST_DB_URL = get_db_url(TEST_DB_NAME)

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

from .base_embedder import BaseEmbedder
from .mock_embedder import MockEmbedder

__all__ = ['BaseEmbedder', 'MockEmbedder']

```
### src/rag/embeddings/mock_embedder.py

```python
"""Mock embedder for testing."""

import random
from typing import List
from .base_embedder import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder that returns random vectors."""

    def __init__(self, dimension: int = 1536):
        """Initialize mock embedder.

        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return [[random.random() for _ in range(self.dimension)] for _ in texts]

```
### src/rag/embeddings/base_embedder.py

```python
"""Base embedder interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Base class for embedding generators."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings generated by this embedder.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        pass

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
## src/rag/chunking

### src/rag/chunking/__init__.py

```python
"""Text chunking utilities."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents a chunk of text with its position in the original document."""
    
    content: str  # Changed from text to content to match the database model
    start: int
    end: int


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[TextChunk]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # Adjust end to not split words
        if end < len(text):
            # Look for the last space before chunk_size
            while end > start and text[end] != ' ':
                end -= 1
            # If no space found, force split at chunk_size
            if end == start:
                end = start + chunk_size
        else:
            end = len(text)
            
        # Create chunk
        chunk = TextChunk(
            content=text[start:end].strip(),  # Changed from text to content
            start=start,
            end=end
        )
        chunks.append(chunk)
        
        # Calculate next start position with overlap
        start = end - overlap
        if start < 0:
            start = 0
            
    return chunks

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
    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    lines = text.splitlines()
    if not lines:
        return [text]  # Return original text if empty or just whitespace
        
    chunks = []
    start = 0

    while start < len(lines):
        # Calculate end of current chunk
        end = min(start + chunk_size, len(lines))
        
        # Join lines for this chunk
        chunk = '\n'.join(lines[start:end])
        chunks.append(chunk)
        
        # If we've reached the end, break
        if end == len(lines):
            break
            
        # Move start position, accounting for overlap
        start = end - overlap

    return chunks

```
### src/rag/db/models.py

```python
"""Database models for RAG."""

from datetime import datetime, timezone as tz
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, BigInteger, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

# Use the new SQLAlchemy 2.0 style declarative base
Base = declarative_base()


def utc_now():
    """Get current UTC datetime."""
    return datetime.now(tz.utc)


class Project(Base):
    """Project model."""

    __tablename__ = 'projects'
    __table_args__ = (
        UniqueConstraint('name', name='uix_project_name'),
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now)

    files = relationship('File', back_populates='project', cascade='all, delete-orphan')


class File(Base):
    """File model."""

    __tablename__ = 'files'
    __table_args__ = (
        UniqueConstraint('filename', 'file_path', 'project_id', name='uix_file_path_project'),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    crc = Column(BigInteger, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    project = relationship('Project', back_populates='files')
    chunks = relationship('Chunk', back_populates='file', cascade='all, delete-orphan')


class Chunk(Base):
    """Chunk model with vector embedding."""

    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'))
    content = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    embedding = Column(Vector(1536))  # Default to OpenAI's dimension
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

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
"""Database file handler."""

import os
import hashlib
import logging
import zlib
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from .models import Base, Project, File, Chunk
from ..embeddings.base_embedder import BaseEmbedder
from ..chunking import split_text_into_chunks

logger = logging.getLogger(__name__)

class DBFileHandler:
    """Database file handler."""

    def __init__(self, db_url: str, embedder: BaseEmbedder):
        """Initialize database file handler."""
        self.engine = create_engine(
            db_url,
            poolclass=NullPool,
            connect_args={"connect_timeout": 5}
        )
        self.Session = sessionmaker(bind=self.engine)
        self.embedder = embedder

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
        return zlib.crc32(file_content.encode('utf-8'))

    def get_or_create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Get existing project or create a new one."""
        with self.session_scope() as session:
            project = session.query(Project).filter(Project.name == name).first()
            if not project:
                project = Project(
                    name=name,
                    description=description,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(project)
                session.flush()
            
            # Create a detached copy of the project
            detached_project = Project(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at
            )
            return detached_project

    def process_file(
        self, 
        project_id: int, 
        file_path: str, 
        content: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> Tuple[File, bool]:
        """
        Process a file and store its chunks with embeddings.
        Returns tuple of (file, is_updated) where is_updated indicates if an existing file was updated.
        """
        filename = os.path.basename(file_path)
        file_type = os.path.splitext(filename)[1]
        current_crc = self.calculate_crc(content)
        file_size = len(content.encode('utf-8'))  # Calculate file size in bytes
        
        with self.session_scope() as session:
            # Check if file exists
            existing_file = session.query(File).filter(
                File.filename == filename,
                File.file_path == file_path,
                File.project_id == project_id
            ).first()
            
            is_updated = False
            if existing_file:
                if existing_file.crc != current_crc:
                    logger.info(f"File {file_path} has changed, updating chunks...")
                    # Delete existing chunks
                    session.query(Chunk).filter(Chunk.file_id == existing_file.id).delete()
                    # Update file
                    existing_file.crc = current_crc
                    existing_file.file_size = file_size  # Update file size
                    existing_file.updated_at = datetime.now(timezone.utc)
                    file = existing_file
                    is_updated = True
                else:
                    logger.info(f"File {file_path} unchanged, skipping...")
                    # Create a detached copy of the file
                    detached_file = File(
                        id=existing_file.id,
                        filename=existing_file.filename,
                        file_path=existing_file.file_path,
                        file_type=existing_file.file_type,
                        crc=existing_file.crc,
                        file_size=existing_file.file_size,  # Include file size
                        project_id=existing_file.project_id,
                        created_at=existing_file.created_at,
                        updated_at=existing_file.updated_at
                    )
                    return detached_file, False
            else:
                logger.info(f"Creating new file entry for {file_path}")
                file = File(
                    filename=filename,
                    file_path=file_path,
                    file_type=file_type,
                    crc=current_crc,
                    file_size=file_size,  # Set file size for new files
                    project_id=project_id,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(file)
                session.flush()
            
            # Process chunks
            chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
            embeddings = self.embedder.embed_documents([chunk.content for chunk in chunks])
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db_chunk = Chunk(
                    content=chunk.content,
                    start_char=chunk.start,
                    end_char=chunk.end,
                    embedding=embedding,
                    file_id=file.id,
                    chunk_index=idx,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(db_chunk)
            
            session.flush()
            
            # Create a detached copy of the file
            detached_file = File(
                id=file.id,
                filename=file.filename,
                file_path=file.file_path,
                file_type=file.file_type,
                crc=file.crc,
                file_size=file.file_size,  # Include file size
                project_id=file.project_id,
                created_at=file.created_at,
                updated_at=file.updated_at
            )
            return detached_file, is_updated

```
## src/scripts

### src/scripts/run_example.py

```python
#!/usr/bin/env python3
"""Example script demonstrating file ingestion and embedding."""

import os
import sys
import logging
from pathlib import Path

from rag.config import get_db_url, OPENAI_API_KEY
from rag.embeddings.mock_embedder import MockEmbedder
from rag.embeddings.openai_embedder import OpenAIEmbedder
from rag.db.db_file_handler import DBFileHandler
from rag.db.models import Base
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_embedder(embedder_type="mock"):
    """Create an embedder instance.
    
    Args:
        embedder_type: Type of embedder to create ('mock' or 'openai')
    
    Returns:
        Embedder instance
    """
    if embedder_type == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment")
        return OpenAIEmbedder(api_key=OPENAI_API_KEY)
    return MockEmbedder()

def ensure_tables_exist():
    """Ensure database tables exist."""
    engine = create_engine(get_db_url())
    Base.metadata.create_all(engine)
    logger.info("Database tables created")

def ingest_file(file_path: str, embedder_type: str = "mock"):
    """Ingest a file into the database.
    
    Args:
        file_path: Path to the file to ingest
        embedder_type: Type of embedder to use
    """
    # Ensure tables exist
    ensure_tables_exist()
    
    # Create embedder
    embedder = create_embedder(embedder_type)
    logger.info(f"Using {embedder.__class__.__name__}")

    # Create DB handler
    handler = DBFileHandler(get_db_url(), embedder)
    
    # Create or get project
    project = handler.get_or_create_project("Demo Project", "Example file ingestion")
    logger.info(f"Using project: {project.name} (ID: {project.id})")
    
    # Add file to project
    file = handler.add_file(project.id, file_path)
    logger.info(f"Added file: {file.filename} (ID: {file.id})")
    
    # Print chunk count
    with handler.session_scope() as session:
        chunk_count = session.query(handler.Chunk).filter_by(file_id=file.id).count()
        logger.info(f"Created {chunk_count} chunks with embeddings")

def main():
    """Main entry point."""
    # Check if a file was provided
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <file_path> [embedder_type]")
        print("embedder_type can be 'mock' or 'openai'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    embedder_type = sys.argv[2] if len(sys.argv) > 2 else "mock"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    try:
        ingest_file(file_path, embedder_type)
        logger.info("File ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error during file ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

```
