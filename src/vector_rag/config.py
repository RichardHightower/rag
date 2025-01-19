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
    """Get database URL with optional database name override.

    Args:
        dbname: Database name to use, defaults to DB_NAME

    Returns:
        str: Complete database URL
    """
    db = dbname or DB_NAME
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db}"


DB_URL = get_db_url()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# Vector dimensions and search configuration
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # OpenAI's default
VECTOR_INDEX_LISTS = int(
    os.getenv("VECTOR_INDEX_LISTS", "100")
)  # Number of IVFFlat lists
VECTOR_INDEX_PROBES = int(
    os.getenv("VECTOR_INDEX_PROBES", "10")
)  # Number of probes for search
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))

# Text chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# File processing configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
SUPPORTED_FILE_TYPES = os.getenv(
    "SUPPORTED_FILE_TYPES", "txt,md,py,js,jsx,ts,tsx,html,css,json,yaml,yml"
).split(",")

# Test configuration
TEST_DB_NAME = "vectordb_test"
TEST_DB_URL = get_db_url(TEST_DB_NAME)

# Batch processing configuration
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))

# Search configuration
DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "10"))
MAX_PAGE_SIZE = int(os.getenv("MAX_PAGE_SIZE", "100"))

# Performance tuning
POOL_SIZE = int(os.getenv("POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("POOL_RECYCLE", "1800"))  # 30 minutes

# Path configuration
LOG_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "tmp"

# Ensure required directories exist
LOG_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
