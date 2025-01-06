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
- Install the package in editable mode with dev dependencies
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
- **Sentence Transformers**: Alternative for generating embeddings

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

### Development Tools
- **black**: Code formatting
- **mypy**: Static type checking
- **isort**: Import sorting

## Task Commands Reference

### Development Setup and Maintenance

```bash
task setup-dev             # Initial dev environment setup (venv, dependencies)
task verify-deps           # Verify dependencies are correctly installed
task freeze               # Generate requirements.txt with top-level dependencies
```

### Testing

The project provides several test commands for different testing scenarios:

```bash
task test:all             # Run all Python tests across all test directories
                         # This is the most comprehensive test command
                         # Use this to verify everything works before commits

task test:name -- <pattern>  # Run tests matching a specific pattern
                            # Examples:
                            # Run a specific test function:
                            #   task test:name -- test_basic_chunking
                            # Run all tests in a module:
                            #   task test:name -- test_line_chunker
                            # Run tests matching a pattern:
                            #   task test:name -- "chunker.*basic"

task test:integration    # Run only integration tests (in tests/integration/)
                        # These tests interact with the database
                        # Will wait 5 seconds for DB to start before running
                        # Use -v flag for verbose output

task test:coverage      # Run tests with coverage reporting
                        # Shows line-by-line coverage information
                        # Reports missing coverage in the terminal
                        # Essential to run before submitting PRs
```

### Pre-Commit Requirements

Before committing code or submitting pull requests, you should run:

1. `task lint` - This runs:
   - Code formatting (black, isort)
   - Type checking (mypy)
   - All tests (test:all)
   This ensures your code meets style guidelines and passes all tests.

2. `task test:coverage` - This checks test coverage and reports:
   - Percentage of code covered by tests
   - Which lines are not covered
   - Helps identify areas needing additional tests
   
Example pre-commit workflow:
```bash
# Format and verify code
task lint

# Check test coverage
task test:coverage

# If all checks pass, commit your changes
git commit -m "Your commit message"
```

### Testing Best Practices

1. **Running Specific Tests**
   - Use `test:name` for focused testing during development
   - Always run `test:all` before committing
   - Run `test:integration` when changing database interactions

2. **Coverage Requirements**
   - Aim for high test coverage (>80%)
   - Run `test:coverage` to identify gaps
   - Write tests for any uncovered code

3. **Integration Testing**
   - Database should be running (`task db:up`)
   - Uses a separate test database
   - Automatically handles test data cleanup

### Code Quality

```bash
task format              # Format code with black and isort
task typecheck          # Run mypy type checking
task lint               # Run all code quality checks (format, typecheck, test:all)
```

### Documentation

```bash
task documentation:create-project-markdown  # Create Markdown for LLMs
```

### Database Management

```bash
task db:recreate         # Recreate database from scratch
task psql               # Start interactive psql session
```

### Demo and Examples

```bash
task demo:mock           # Run example ingestion with mock embedder
task demo:openai         # Run example ingestion with OpenAI embedder
```

## Development Workflow

1. **Starting Development**
   - Start database: `task db:up`
   - Verify setup: `task verify-deps`

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
   - Run all tests: `task test:all`
   - Run specific test: `task test:name -- test_name`
   - Run integration tests: `task test:integration`
   - Check coverage: `task test:coverage`

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
- Run with verbose output: `task test:name -- -v test_name`
- Run specific test file: `task test:name -- "test_file.py"`
- Debug specific test: `task test:name -- -s test_name`

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
        "openai>=1.58.1,<2.0.0",
        "sqlalchemy>=2.0.36,<3.0.0",
        "psycopg2-binary>=2.9.10,<3.0.0",
        "pgvector>=0.3.6,<0.4.0",
        "python-dotenv>=1.0.1,<2.0.0",
    ],
    extras_require={
        "dev": [
            "black>=24.1.0,<25.0.0",
            "isort>=5.13.0,<6.0.0",
            "mypy>=1.8.0,<2.0.0",
            "pytest>=8.0.0,<9.0.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "types-psycopg2>=2.9.21,<3.0.0"
        ]
    },
    python_requires=">=3.12,<3.13",
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
from rag.db import Chunk
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
    chunks = [Chunk(content=text) for text in texts]
    embeddings = embedder.embed_texts(chunks)

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
    embeddings = embedder.embed_texts([Chunk(content=text) for text in texts])

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
from rag.model import File


@pytest.fixture
def sample_file():
    """Create a temporary sample file."""
    content = "This is a test file.\nIt has multiple lines.\nEach line will be chunked."
    return File(
        name="test.txt", path="/path/to/test.txt", crc="test123", content=content
    )


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
    file = handler.add_file(project.id, sample_file)
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

#### tests/chunking/

##### test_line_chunker.py

```python
"""Test text chunking utilities."""

import pytest

from rag.chunking import LineChunker
from rag.model import File


@pytest.fixture
def chunker():
    """Create a line chunker instance."""
    return LineChunker()


@pytest.fixture
def sample_file():
    """Create a sample file for testing."""
    return File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="abcdef123456",
        content="\n".join([f"Line {i}" for i in range(10)]),
        meta_data={"type": "test"},
    )


def test_basic_chunking(chunker, sample_file):
    """Test basic text chunking with default parameters."""
    # Default chunk_size=500, overlap=50 should return single chunk
    chunks = chunker.chunk_text(sample_file)
    assert len(chunks) == 1
    assert chunks[0].content == sample_file.content
    assert chunks[0].index == 0


def test_custom_chunk_size(chunker, sample_file):
    """Test chunking with custom chunk size."""
    # Set chunk_size to 4 lines, overlap to 1
    chunker = LineChunker(chunk_size=4, overlap=1)
    chunks = chunker.chunk_text(sample_file)

    assert len(chunks) == 3
    # Verify first chunk
    assert chunks[0].content == "\n".join(["Line 0", "Line 1", "Line 2", "Line 3"])
    assert chunks[0].index == 0

    # Verify middle chunk has overlap
    assert chunks[1].content == "\n".join(["Line 3", "Line 4", "Line 5", "Line 6"])
    assert chunks[1].index == 1

    # Verify last chunk
    assert chunks[2].content == "\n".join(["Line 6", "Line 7", "Line 8", "Line 9"])
    assert chunks[2].index == 2


def test_empty_text(chunker):
    """Test chunking empty text."""
    empty_file = File(
        name="empty.txt",
        path="/path/to/empty.txt",
        crc="empty123",
        content="",
        meta_data={},
    )
    chunks = chunker.chunk_text(empty_file)
    assert len(chunks) == 1
    assert chunks[0].content == ""


def test_whitespace_text(chunker):
    """Test chunking whitespace text."""
    whitespace_file = File(
        name="whitespace.txt",
        path="/path/to/whitespace.txt",
        crc="space123",
        content="   \n  \n  ",
        meta_data={},
    )
    chunks = chunker.chunk_text(whitespace_file)
    assert len(chunks) == 1
    assert chunks[0].content == "   \n  \n  "


def test_single_line(chunker):
    """Test chunking single line of text."""
    single_line_file = File(
        name="single.txt",
        path="/path/to/single.txt",
        crc="single123",
        content="Single line",
        meta_data={},
    )
    chunks = chunker.chunk_text(single_line_file)
    assert len(chunks) == 1
    assert chunks[0].content == "Single line"


def test_text_smaller_than_chunk(chunker):
    """Test when text is smaller than chunk size."""
    small_file = File(
        name="small.txt",
        path="/path/to/small.txt",
        crc="small123",
        content="\n".join([f"Line {i}" for i in range(5)]),
        meta_data={},
    )
    chunker = LineChunker(chunk_size=10, overlap=5)
    chunks = chunker.chunk_text(small_file)
    assert len(chunks) == 1
    assert chunks[0].content == small_file.content


def test_no_overlap(chunker):
    """Test chunking with no overlap."""
    file = File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="test123",
        content="\n".join([f"Line {i}" for i in range(6)]),
        meta_data={},
    )
    chunker = LineChunker(chunk_size=2, overlap=0)
    chunks = chunker.chunk_text(file)

    assert len(chunks) == 3
    assert chunks[0].content == "\n".join(["Line 0", "Line 1"])
    assert chunks[1].content == "\n".join(["Line 2", "Line 3"])
    assert chunks[2].content == "\n".join(["Line 4", "Line 5"])


def test_invalid_chunk_size(chunker, sample_file):
    """Test invalid chunk size."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker = LineChunker(chunk_size=0)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker = LineChunker(chunk_size=-1)
        chunker.chunk_text(sample_file)


def test_invalid_overlap(chunker, sample_file):
    """Test invalid overlap size."""
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunker = LineChunker(chunk_size=2, overlap=-1)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunker = LineChunker(chunk_size=2, overlap=2)
        chunker.chunk_text(sample_file)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunker = LineChunker(chunk_size=2, overlap=3)
        chunker.chunk_text(sample_file)

```

#### tests/db/

##### test_db_file_handler.py

```python
"""Test database file handler."""

import hashlib
import os
import tempfile
from logging import debug
from pathlib import Path

import pytest

from rag.config import TEST_DB_NAME, get_db_url
from rag.db.db_file_handler import DBFileHandler
from rag.embeddings.mock_embedder import MockEmbedder
from rag.model import File as FileModel


@pytest.fixture
def embedder():
    """Create a mock embedder for testing."""
    return MockEmbedder()


def create_test_file(content="Line 1\nLine 2\nLine 3\n"):
    """Create a FileModel instance for testing."""
    return FileModel(
        name="test.txt",
        path="/path/to/test.txt",
        crc=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content=content,
        meta_data={},
    )


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


def test_add_file(test_db, embedder):
    """Test adding a file to a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project
    project = handler.create_project("Test Project")

    # Create test file
    file_model = create_test_file()

    debug(file_model)

    # Add file
    success = handler.add_file(project.id, file_model)
    assert success is not None

    # Verify file was added
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        assert file.project_id == project.id
        assert file.filename == file_model.name
        assert file.file_path == file_model.path
        assert file.created_at is not None

        # Verify chunks were created
        chunks = session.query(handler.Chunk).filter_by(file_id=file.id).all()
        assert len(chunks) > 0

        # Verify chunk content and embeddings
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None
            assert len(chunk.embedding) == embedder.get_dimension()


def test_add_file_to_nonexistent_project(test_db, embedder):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)
    file_model = create_test_file()

    # Try to add file to non-existent project
    success = handler.add_file(999, file_model)
    assert success is None


def test_remove_file(test_db, embedder):
    """Test removing a file from a project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project and add file
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    success = handler.add_file(project.id, file_model)
    assert success is not None

    # Get file ID
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id

    # Remove file
    assert handler.remove_file(project.id, file_id) is True

    # Verify file is removed
    with handler.session_scope() as session:
        assert session.get(handler.File, file_id) is None


def test_remove_nonexistent_file(test_db, embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create project
    project = handler.create_project("Test Project")

    # Try to remove non-existent file
    assert handler.remove_file(project.id, 999) is False


def test_remove_file_wrong_project(test_db, embedder):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler(get_db_url(TEST_DB_NAME), embedder)

    # Create two projects
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")

    # Add file to project1
    file_model = create_test_file()
    file = handler.add_file(project1.id, file_model)
    assert file is not None

    # Get file ID
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id

    # Try to remove file from project2
    assert handler.remove_file(project2.id, file_id) is False


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
def get_db_url(dbname: str = "vectordb") -> str:
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

##### model.py

```python
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

from rag.model import Chunk


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
    def embed_texts(self, texts: List[Chunk]) -> List[List[float]]:
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

from ..model import Chunk
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

    def embed_texts(self, texts: List[Chunk]) -> List[List[float]]:
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

from typing import List, Optional

import openai

from ..config import EMBEDDING_DIM, OPENAI_API_KEY, OPENAI_MODEL
from ..model import Chunk
from .base import Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI embedder implementation."""

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        dimension: int = EMBEDDING_DIM,
        api_key: Optional[str] = OPENAI_API_KEY,
        batch_size: int = 16,
    ):
        """Initialize OpenAI embedder.

        Args:
            model_name: Name of the OpenAI model to use
            dimension: Dimension of the embeddings
            api_key: OpenAI API key
            batch_size: Number of texts to embed in one batch

        Raises:
            ValueError: If no API key is provided
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

    def embed_texts(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API.

        Args:
            chunks: List of text chunks to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = [chunk.content for chunk in chunks[i : i + self.batch_size]]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

```

##### src/rag/chunking/

###### __init__.py

```python
"""Tools to break files into chunks."""

from .line_chunker import LineChunker

```

###### base_chunker.py

```python
"""Base chunker interface."""

from abc import ABC, abstractmethod
from typing import List

from ..model import Chunk, File


class Chunker(ABC):
    """Abstract base class for text chunking implementations."""

    @abstractmethod
    def chunk_text(self, file: File) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            file: File to split
            chunk_size: Number of lines per chunk
            overlap: Number of lines to overlap between chunks

        Returns:
            List[Chunk]: List of text chunks
        """
        pass

```

###### line_chunker.py

```python
from typing import List

from ..config import CHUNK_OVERLAP, CHUNK_SIZE
from ..model import Chunk, File
from .base_chunker import Chunker


class LineChunker(Chunker):
    """Chunker that splits text based on lines."""

    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Validate inputs
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    def chunk_text(self, file: File) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            file: File to split


        Returns:
            List[Chunk]: List of text chunks
        """

        lines = file.content.splitlines()
        if not lines:
            return [Chunk(target_size=self.chunk_size, content=file.content, index=0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(lines):
            # Calculate end of current chunk
            end = min(start + self.chunk_size, len(lines))

            # Join lines for this chunk
            chunk_content = "\n".join(lines[start:end])
            chunks.append(
                Chunk(
                    target_size=self.chunk_size,
                    content=chunk_content,
                    index=chunk_index,
                )
            )

            # If we've reached the end, break
            if end == len(lines):
                break

            # Move start position, accounting for overlap
            start = end - self.overlap
            chunk_index += 1

        return chunks

```

##### src/rag/db/

###### __init__.py

```python
"""Database package for RAG."""

from .db_file_handler import DBFileHandler
from .dimension_utils import ensure_vector_dimension
from .models import Chunk, File, Project

__all__ = ["Project", "File", "Chunk", "ensure_vector_dimension", "DBFileHandler"]

```

###### db_file_handler.py

```python
"""Database file handler for managing projects and files."""

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
        """Add a file to a project.

        Args:
            project_id: ID of the project

        Returns:
            bool: Was the file created or not
        """

        with self.session_scope() as session:
            # Verify project exists
            project = session.get(Project, project_id)

            if not project:
                # TODO turn this into an exception
                return None

            # TODO Check to see if the file already exists with the same name, path, crc and project id in the DB,
            # if it does, return false. We won't reindex files that already exist.

            # Create file record
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
from typing import List, Optional

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import (BigInteger, Column, DateTime, ForeignKey, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(tz.utc)


class Project(Base):
    """Project model."""

    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("name", name="uix_project_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    files: Mapped[List["File"]] = relationship(
        "File", back_populates="project", cascade="all, delete-orphan"
    )


class File(Base):
    """File model."""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE")
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    crc: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    last_ingested: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    project: Mapped["Project"] = relationship("Project", back_populates="files")
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="file", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Chunk model with vector embedding."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))  # type: ignore
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    file: Mapped["File"] = relationship("File", back_populates="chunks")

```

#### src/scripts/

##### run_example.py

```python
#!/usr/bin/env python3
"""Example script demonstrating file ingestion and embedding."""

import hashlib
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
from rag.model import File as FileModel

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


def create_file_model(file_path: str) -> FileModel:
    """Create a FileModel instance from a file path.

    Args:
        file_path: Path to the file

    Returns:
        FileModel instance
    """
    path = Path(file_path)
    content = path.read_text()
    crc = hashlib.md5(content.encode()).hexdigest()

    return FileModel(
        name=path.name,
        path=str(path),
        crc=crc,
        content=content,
        meta_data={"type": path.suffix.lstrip(".")},
    )


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
    handler = DBFileHandler(embedder=embedder)

    # Create or get project
    project = handler.get_or_create_project("Demo Project", "Example file ingestion")
    logger.info(f"Using project: {project.name} (ID: {project.id})")

    # Create FileModel
    file_model = create_file_model(file_path)

    # Add file to project
    success = handler.add_file(project.id, file_model)
    if not success:
        logger.error("Failed to add file")
        return

    logger.info(f"Added file: {file_model.name}")

    # Print chunk count
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        if file:
            chunk_count = (
                session.query(handler.Chunk).filter_by(file_id=file.id).count()
            )
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
