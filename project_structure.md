# rag

## ./

### README.md

```markdown
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

```

### developer_guide.md

```markdown
# Developer Guide

This guide provides detailed information for developers working on the RAG project.

## Development Environment Setup

### Prerequisites

- Python 3.13+
- Docker and Docker Compose
- Task (task runner)
- PostgreSQL client (for psql)

### Initial Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag
```

2. Set up development environment:
```bash
task setup-dev
```
This command will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings:
# - POSTGRES_USER
# - POSTGRES_PASSWORD
# - POSTGRES_DB
# - POSTGRES_HOST
# - POSTGRES_PORT
# - OPENAI_API_KEY (if using OpenAI embeddings)
```

## Core Libraries

### Database
- **SQLAlchemy**: ORM for database interactions
- **pgvector**: PostgreSQL extension for vector similarity search
- **psycopg2**: PostgreSQL adapter for Python

### Machine Learning
- **OpenAI**: For generating embeddings (optional)
- **Hugging Face Transformers**: Alternative for generating embeddings

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

### Development Tools
- **black**: Code formatting
- **mypy**: Static type checking
- **ruff**: Fast Python linter

## Docker Management

### Database Container

Start the database:
```bash
task db:up
```

Verify database setup:
```bash
task db:test
```

Other database commands:
```bash
task db:down            # Stop database
task db:recreate        # Reset database
task db:psql           # Open PostgreSQL console
task db:list-tables    # List all tables
```

## Task Commands Reference

### Development Workflow

```bash
task setup-dev         # Initial dev environment setup
task verify-deps       # Verify dependencies are correctly installed
task freeze           # Update requirements.txt
```

### Code Quality

```bash
task format           # Format code with black
task lint            # Run all linters
task typecheck       # Run mypy type checking
```

### Testing

```bash
task test:all         # Run all tests
task test:integration # Run integration tests only
task test:py         # Run Python unit tests
task coverage:py     # Run tests with coverage report
```

### Demo and Examples

```bash
task demo:mock        # Run demo with mock embedder
task demo:openai      # Run demo with OpenAI embedder
```

### Database Management

```bash
task db:build              # Build custom Postgres image
task db:up                 # Start database
task db:down              # Stop database
task db:create-tables     # Initialize schema
task db:recreate          # Reset database
task psql                 # Start psql session
```

## Development Workflow

1. **Starting Development**
   - Start database: `task db:up`
   - Verify setup: `task db:test`

2. **Making Changes**
   - Write code
   - Format: `task format`
   - Type check: `task typecheck`
   - Run tests: `task test:all`

3. **Database Changes**
   - Edit models in `src/rag/db/models.py`
   - Update schema in `db/sql/ddl/`
   - Recreate database: `task db:recreate`

4. **Testing Changes**
   - Add tests in `tests/`
   - Run specific test file: `pytest tests/path/to/test.py`
   - Check coverage: `task coverage:py`

## Troubleshooting

### Database Issues
- Verify database is running: `docker ps`
- Check logs: `docker logs rag-db-1`
- Reset database: `task db:recreate`

### Environment Issues
- Verify dependencies: `task verify-deps`
- Recreate virtual environment:
  ```bash
  rm -rf venv
  task setup-dev
  ```

### Testing Issues
- Run with verbose output: `pytest -vv`
- Debug specific test: `pytest tests/path/to/test.py -k test_name -s`

## Best Practices

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Run `task format` before committing

2. **Testing**
   - Write tests for new features
   - Maintain high coverage
   - Use fixtures for common setup

3. **Database**
   - Use SQLAlchemy for database operations
   - Add indexes for frequently queried fields
   - Keep vector dimensions consistent

4. **Documentation**
   - Update docstrings
   - Keep README.md current
   - Document complex algorithms

```

### project_structure.md

```markdown
# rag

## ./


```

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

### tests/

#### conftest.py

```python
"""Pytest configuration file."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from rag.config import get_db_url
from rag.embeddings import MockEmbedder


def create_database():
    """Create a test database if it doesn't exist."""
    template_db_url = get_db_url("postgres")
    test_db_url = get_db_url("vectordb_test")

    engine = create_engine(template_db_url)
    try:
        with engine.connect() as conn:
            # Disconnect all users from the database we will drop
            conn.execute(
                text(
                    f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = 'vectordb_test'
                AND pid <> pg_backend_pid()
                """
                )
            )
            conn.execute(text("commit"))

            # Drop and recreate the database
            conn.execute(text("DROP DATABASE IF EXISTS vectordb_test"))
            conn.execute(text("commit"))
            conn.execute(text("CREATE DATABASE vectordb_test"))
            conn.execute(text("commit"))

            # Enable pgvector extension
            test_engine = create_engine(test_db_url)
            with test_engine.connect() as test_conn:
                test_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                test_conn.execute(text("commit"))
    except OperationalError as e:
        pytest.fail(
            f"Failed to connect to database. Please check your environment variables and ensure the database is running: {e}"
        )


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder(dimension=4)  # Small dimension for testing


@pytest.fixture(scope="function")
def test_db():
    """Create a test database."""
    # Connect to default postgres database to create/drop test database
    default_engine = create_engine(
        f"postgresql://postgres:postgres@localhost:5433/postgres"
    )

    test_db_name = "vectordb_test"

    # Drop test database if it exists and create it fresh
    with default_engine.connect() as conn:
        # Terminate existing connections
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{test_db_name}'
            AND pid <> pg_backend_pid()
            """
            )
        )
        conn.execute(text("commit"))

        # Drop and recreate database
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name}"))
        conn.execute(text("commit"))
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))
        conn.execute(text("commit"))

    # Create engine for test database
    test_engine = create_engine(get_db_url(test_db_name))

    # Create vector extension and tables
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("commit"))

        # Import and create tables
        from rag.db.models import Base

        Base.metadata.create_all(test_engine)
        conn.execute(text("commit"))

    yield test_engine

    # Cleanup: drop test database
    with default_engine.connect() as conn:
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{test_db_name}'
            AND pid <> pg_backend_pid()
            """
            )
        )
        conn.execute(text("commit"))
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name}"))
        conn.execute(text("commit"))

```

#### tests/unit/

#### tests/embeddings/

##### test_embedders.py

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

##### test_openai_embedder.py

```python
"""Test OpenAI embedder."""

import os
from unittest.mock import Mock, patch

import pytest

from rag.config import EMBEDDING_DIM, OPENAI_MODEL
from rag.embeddings.openai_embedder import OpenAIEmbedder


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
        model_name="custom-model", dimension=128, api_key="test_key", batch_size=32
    )
    assert embedder.model_name == "custom-model"
    assert embedder.dimension == 128
    assert embedder.batch_size == 32


def test_get_dimension():
    """Test get_dimension method."""
    embedder = OpenAIEmbedder(api_key="test_key", dimension=256)
    assert embedder.get_dimension() == 256


@patch("openai.OpenAI")
def test_embed_texts_single_batch(mock_openai):
    """Test embedding texts that fit in a single batch."""
    # Mock response from OpenAI
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
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
        model=OPENAI_MODEL, input=texts
    )


@patch("openai.OpenAI")
def test_embed_texts_multiple_batches(mock_openai):
    """Test embedding texts that require multiple batches."""
    # Mock responses for two batches
    mock_response1 = Mock()
    mock_response1.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
    mock_response2 = Mock()
    mock_response2.data = [Mock(embedding=[0.5, 0.6])]

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
        model=OPENAI_MODEL, input=["Text1", "Text2"]
    )
    mock_client.embeddings.create.assert_any_call(model=OPENAI_MODEL, input=["Text3"])

```

#### tests/integration/

##### test_ingestion.py

```python
"""Integration test for file ingestion."""

import os
import tempfile
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from rag.config import get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder


@pytest.fixture
def sample_file():
    """Create a temporary sample file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "This is a test file.\nIt has multiple lines.\nEach line will be chunked."
        )
        return Path(f.name)


@pytest.fixture
def unique_name():
    """Generate a unique project name."""
    return f"Test Project {uuid.uuid4()}"


def test_file_ingestion(test_db, sample_file, unique_name):
    """Test complete file ingestion flow."""
    # Create handler with mock embedder
    handler = DBFileHandler(get_db_url(), MockEmbedder())

    # Create project
    project = handler.create_project(unique_name)
    assert project is not None
    assert project.name == unique_name

    # Add file
    file = handler.add_file(project.id, str(sample_file))
    assert file is not None

    # Verify chunks were created
    with handler.session_scope() as session:
        # Count chunks
        chunk_count = session.query(handler.Chunk).filter_by(file_id=file.id).count()
        assert chunk_count > 0

        # Verify embeddings exist
        chunks = session.query(handler.Chunk).filter_by(file_id=file.id).all()
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == MockEmbedder().get_dimension()


def test_project_uniqueness(test_db, unique_name):
    """Test project name uniqueness constraints."""
    handler = DBFileHandler(get_db_url(), MockEmbedder())

    # Create initial project
    project1 = handler.create_project(unique_name)
    assert project1 is not None
    assert project1.name == unique_name

    # Try to create another project with the same name
    with pytest.raises(
        ValueError, match=f"Project with name '{unique_name}' already exists"
    ):
        handler.create_project(unique_name)

    # Test get_or_create_project
    project2 = handler.get_or_create_project(unique_name)
    assert project2.id == project1.id
    assert project2.name == unique_name

    # Test description update
    new_description = "Updated description"
    project3 = handler.get_or_create_project(unique_name, new_description)
    assert project3.id == project1.id
    assert project3.name == unique_name
    assert project3.description == new_description


def test_multiple_projects(test_db):
    """Test creating multiple projects with different names."""
    handler = DBFileHandler(get_db_url(), MockEmbedder())

    # Create multiple projects
    names = [f"Project {i}" for i in range(3)]
    projects = []

    for name in names:
        project = handler.create_project(name)
        assert project is not None
        assert project.name == name
        projects.append(project)

    # Verify all projects exist and have unique IDs
    project_ids = {p.id for p in projects}
    assert len(project_ids) == len(names)

    for project in projects:
        handler.delete_project(project.id)


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up any .txt files in the current directory
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass

```

#### tests/db/

##### test_chunking.py

```python
"""Test text chunking utilities."""

import pytest

from rag.db.chunking import chunk_text


def test_basic_chunking():
    """Test basic text chunking with default parameters."""
    # Create test text with 10 lines
    text = "\n".join([f"Line {i}" for i in range(10)])

    # Default chunk_size=500, overlap=50 should return single chunk
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_custom_chunk_size():
    """Test chunking with custom chunk size."""
    # Create test text with 10 lines
    lines = [f"Line {i}" for i in range(10)]
    text = "\n".join(lines)

    # Set chunk_size to 4 lines, overlap to 1
    chunks = chunk_text(text, chunk_size=4, overlap=1)

    # Expected chunks with overlap:
    # Chunk 1: lines 0-3
    # Chunk 2: lines 3-6
    # Chunk 3: lines 6-9
    assert len(chunks) == 3

    # Verify first chunk
    assert chunks[0] == "\n".join(["Line 0", "Line 1", "Line 2", "Line 3"])

    # Verify middle chunk has overlap
    assert chunks[1] == "\n".join(["Line 3", "Line 4", "Line 5", "Line 6"])

    # Verify last chunk
    assert chunks[2] == "\n".join(["Line 6", "Line 7", "Line 8", "Line 9"])


def test_empty_text():
    """Test chunking empty text."""
    chunks = chunk_text("")
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_whitespace_text():
    """Test chunking whitespace text."""
    chunks = chunk_text("   \n  \n  ")
    assert len(chunks) == 1
    assert chunks[0] == "   \n  \n  "


def test_single_line():
    """Test chunking single line of text."""
    text = "Single line"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_smaller_than_chunk():
    """Test when text is smaller than chunk size."""
    text = "\n".join([f"Line {i}" for i in range(5)])
    chunks = chunk_text(
        text, chunk_size=10, overlap=5
    )  # Make sure overlap < chunk_size
    assert len(chunks) == 1
    assert chunks[0] == text


def test_no_overlap():
    """Test chunking with no overlap."""
    # Create test text with 6 lines
    lines = [f"Line {i}" for i in range(6)]
    text = "\n".join(lines)

    # Set chunk_size to 2 lines, no overlap
    chunks = chunk_text(text, chunk_size=2, overlap=0)

    # Expected chunks:
    # Chunk 1: lines 0-1
    # Chunk 2: lines 2-3
    # Chunk 3: lines 4-5
    assert len(chunks) == 3
    assert chunks[0] == "\n".join(["Line 0", "Line 1"])
    assert chunks[1] == "\n".join(["Line 2", "Line 3"])
    assert chunks[2] == "\n".join(["Line 4", "Line 5"])


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

##### test_db_file_handler.py

```python
"""Test database file handler."""

import os
import tempfile
from pathlib import Path

import pytest

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Line 1\nLine 2\nLine 3\n")
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
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass

```

##### test_dimension_utils.py

```python
"""Test dimension utilities."""

import pytest
from sqlalchemy import text

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.dimension_utils import ensure_vector_dimension


def test_ensure_vector_dimension_same(test_db):
    """Test when current dimension matches desired dimension."""
    engine = test_db

    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """
            )
        )
        conn.commit()

    # Call ensure_vector_dimension with same dimension
    ensure_vector_dimension(engine, 3)

    # Verify dimension is still 3
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """
            )
        )
        assert result.scalar() == 3


def test_ensure_vector_dimension_different(test_db):
    """Test when current dimension differs from desired dimension."""
    engine = test_db

    # Set initial dimension
    with engine.connect() as conn:
        conn.execute(
            text(
                """
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3)
            USING embedding::vector(3);
        """
            )
        )
        conn.commit()

    # Change to different dimension
    ensure_vector_dimension(engine, 5)

    # Verify dimension was changed
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """
            )
        )
        assert result.scalar() == 5

```

### examples/

### db/

#### init.sql

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

```

#### db/sql/

##### db/sql/ddl/

###### 01_init_tables.sql

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

### src/

#### src/rag/

##### __init__.py

```python
"""RAG (Retrieval Augmented Generation) package."""

```

##### config.py

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
def get_db_url(dbname: str = "vector_db") -> str:
    """Get database URL with optional database name override."""
    db = dbname or DB_NAME
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db}"


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

##### src/rag/embeddings/

###### __init__.py

```python
"""Embeddings package for handling text embeddings from various sources."""

from .base import Embedder
from .mock_embedder import MockEmbedder
from .openai_embedder import OpenAIEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "MockEmbedder"]

```

###### base.py

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

###### mock_embedder.py

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
        return [[random.uniform(-1, 1) for _ in range(self.dimension)] for _ in texts]

```

###### openai_embedder.py

```python
"""OpenAI embedder implementation."""

from typing import List

import openai

from ..config import EMBEDDING_DIM, OPENAI_API_KEY, OPENAI_MODEL
from .base import Embedder


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
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

```

##### src/rag/chunking/

##### src/rag/db/

###### __init__.py

```python
"""Database package for RAG."""

from .db_file_handler import DBFileHandler
from .dimension_utils import ensure_vector_dimension
from .models import Chunk, File, Project

__all__ = ["Project", "File", "Chunk", "ensure_vector_dimension", "DBFileHandler"]

```

###### chunking.py

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
        chunk = "\n".join(lines[start:end])
        chunks.append(chunk)

        # If we've reached the end, break
        if end == len(lines):
            break

        # Move start position, accounting for overlap
        start = end - overlap

    return chunks

```

###### db_file_handler.py

```python
"""Database file handler for managing projects and files."""

import hashlib
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, DB_URL
from ..embeddings import Embedder
from .chunking import chunk_text
from .dimension_utils import ensure_vector_dimension
from .models import Base, Chunk, File, Project


class DBFileHandler:
    """Handler for managing files in the database."""

    def __init__(
        self, db_url: Optional[str] = None, embedder: Optional[Embedder] = None
    ):
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

    def add_file(
        self,
        project_id: int,
        file_path: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Optional[File]:
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
            with open(file_path, "r") as f:
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
                file_size=file_size,
            )
            session.add(file)
            session.flush()  # Get file.id

            # Create chunks
            chunks = chunk_text(content, chunk_size, overlap)
            embeddings = self.embedder.embed_texts(chunks)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_obj = Chunk(
                    file_id=file.id, content=chunk, embedding=embedding, chunk_index=idx
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

```

###### dimension_utils.py

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
        result = conn.execute(
            text(
                """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding';
        """
            )
        )
        current_dim = result.scalar()

        if current_dim != desired_dim:
            conn.execute(
                text(
                    f"""
                ALTER TABLE chunks
                ALTER COLUMN embedding TYPE vector({desired_dim})
                USING embedding::vector({desired_dim});
            """
                )
            )
            conn.commit()

```

###### models.py

```python
"""Database models for RAG."""

from datetime import datetime
from datetime import timezone as tz

from pgvector.sqlalchemy import Vector
from sqlalchemy import (BigInteger, Column, DateTime, ForeignKey, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

# Use the new SQLAlchemy 2.0 style declarative base
Base = declarative_base()
Alias = Base

def utc_now():
    """Get current UTC datetime."""
    return datetime.now(tz.utc)


class Project(Base):
    """Project model."""

    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("name", name="uix_project_name"),)

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now)

    files = relationship("File", back_populates="project", cascade="all, delete-orphan")


class File(Base):
    """File model."""

    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    crc = Column(String(32), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    last_updated = Column(DateTime(timezone=True), default=utc_now)
    last_ingested = Column(DateTime(timezone=True), default=utc_now)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    project = relationship("Project", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk model with vector embedding."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Default to OpenAI's dimension
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=utc_now)

    file = relationship("File", back_populates="chunks")

```

#### src/scripts/

##### run_example.py

```python
#!/usr/bin/env python3
"""Example script demonstrating file ingestion and embedding."""

import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine

from rag.config import OPENAI_API_KEY, get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.db.models import Base
from rag.embeddings.mock_embedder import MockEmbedder
from rag.embeddings.openai_embedder import OpenAIEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

