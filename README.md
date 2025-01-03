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
