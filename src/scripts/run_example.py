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
