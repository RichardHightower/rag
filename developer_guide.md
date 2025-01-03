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
