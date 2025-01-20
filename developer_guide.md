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
cd vector_rag
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