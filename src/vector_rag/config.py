"""Configuration module for RAG."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv


class Config:
    """Configuration class for RAG system."""

    def __init__(
        self,
        env_file: Optional[Union[str, Path]] = None,
        DB_USER: Optional[str] = None,
        DB_PASSWORD: Optional[str] = None,
        DB_HOST: Optional[str] = None,
        DB_PORT: Optional[int] = None,
        DB_NAME: Optional[str] = None,
        OPENAI_API_KEY: Optional[str] = None,
        OPENAI_TEXT_EMBED_MODEL: Optional[str] = None,
        LOG_LEVEL: Optional[str] = None,
        LOG_LEVEL_CONSOLE: Optional[str] = None,
        LOG_DIR: Optional[Union[str, Path]] = None,
        CHUNK_SIZE: Optional[int] = None,
        CHUNK_OVERLAP: Optional[int] = None,
        EMBEDDING_DIM: Optional[int] = None,
        **kwargs,
    ):
        """Initialize configuration with optional overrides.

        Args:
            env_file: Path to .env file, defaults to .env in current directory
            LOG_LEVEL: Logging level for file output
            LOG_LEVEL_CONSOLE: Logging level for console output (NONE to disable)
            LOG_DIR: Directory for log files
            **kwargs: Additional configuration overrides
        """
        # Store provided values
        self._config_overrides = {
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_NAME": DB_NAME,
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "OPENAI_TEXT_EMBED_MODEL": OPENAI_TEXT_EMBED_MODEL,
            "LOG_LEVEL": LOG_LEVEL,
            "LOG_LEVEL_CONSOLE": LOG_LEVEL_CONSOLE,
            "LOG_DIR": LOG_DIR,
            "CHUNK_SIZE": CHUNK_SIZE,
            "CHUNK_OVERLAP": CHUNK_OVERLAP,
            "EMBEDDING_DIM": EMBEDDING_DIM,
            **kwargs,
        }

        # Load environment variables
        if env_file:
            self.env_file = Path(env_file)
        else:
            self.env_file = Path.cwd() / ".env"

        if self.env_file.exists():
            load_dotenv(self.env_file)

        # Initialize all configuration values
        self.DB_USER = self.get_or_default("DB_USER", "postgres")
        self.DB_PASSWORD = self.get_or_default("DB_PASSWORD", "postgres")
        self.DB_HOST = self.get_or_default("DB_HOST", "localhost")
        self.DB_PORT = int(self.get_or_default("DB_PORT", "5433"))
        self.DB_NAME = self.get_or_default("DB_NAME", "vectordb")

        # OpenAI configuration
        self.OPENAI_API_KEY = self.get_or_default("OPENAI_API_KEY", "")
        self.OPENAI_TEXT_EMBED_MODEL = self.get_or_default(
            "OPENAI_TEXT_EMBED_MODEL", "text-embedding-3-small"
        )

        # Vector dimensions and search configuration
        self.EMBEDDING_DIM = int(self.get_or_default("EMBEDDING_DIM", "1536"))
        self.VECTOR_INDEX_LISTS = int(self.get_or_default("VECTOR_INDEX_LISTS", "100"))
        self.VECTOR_INDEX_PROBES = int(self.get_or_default("VECTOR_INDEX_PROBES", "10"))
        self.DEFAULT_SIMILARITY_THRESHOLD = float(
            self.get_or_default("DEFAULT_SIMILARITY_THRESHOLD", "0.7")
        )

        # Text chunking configuration
        self.CHUNK_SIZE = int(self.get_or_default("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(self.get_or_default("CHUNK_OVERLAP", "50"))

        # File processing configuration
        self.MAX_FILE_SIZE = int(self.get_or_default("MAX_FILE_SIZE", "10485760"))
        self.SUPPORTED_FILE_TYPES = self.get_or_default(
            "SUPPORTED_FILE_TYPES", "txt,md,py,js,jsx,ts,tsx,html,css,json,yaml,yml"
        ).split(",")

        # Test configuration
        self.TEST_DB_NAME = "vectordb_test"

        # Logging configuration
        self.LOG_LEVEL = self.get_or_default("LOG_LEVEL", "INFO")
        self.LOG_LEVEL_CONSOLE = self.get_or_default("LOG_LEVEL_CONSOLE", "INFO")
        self.LOG_DIR = Path(self.get_or_default("LOG_DIR", Path.cwd() / "logs"))

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with file and optional console output."""
        # Create log directory if it doesn't exist
        self.LOG_DIR.mkdir(exist_ok=True)
        log_file = self.LOG_DIR / "vector_rag.log"

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Capture all levels

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(self._get_log_level(self.LOG_LEVEL))
        logger.addHandler(file_handler)

        # Setup console handler if not disabled
        if self.LOG_LEVEL_CONSOLE.upper() != "NONE":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self._get_log_level(self.LOG_LEVEL_CONSOLE))
            logger.addHandler(console_handler)

        # Log initial configuration
        logging.info(f"Logging initialized. File: {log_file}")
        logging.debug(f"File logging level: {self.LOG_LEVEL}")
        logging.debug(f"Console logging level: {self.LOG_LEVEL_CONSOLE}")

    def _get_log_level(self, level: str) -> int:
        """Convert string log level to logging constant."""
        return getattr(logging, level.upper(), logging.INFO)

    def get_or_default(self, name: str, default_value: Any) -> Any:
        """Get configuration value from various sources.

        Order of precedence:
        1. Constructor arguments
        2. Environment variables
        3. .env file
        4. Default value

        Args:
            name: Name of the configuration value
            default_value: Default value if not found elsewhere

        Returns:
            Configuration value from the highest priority source
        """
        # Check constructor arguments
        if self._config_overrides.get(name) is not None:
            return self._config_overrides[name]

        # Check environment variables
        env_value = os.getenv(name)
        if env_value is not None:
            return env_value

        # Return default if nothing else found
        return default_value

    def get_db_url(self, dbname: Optional[str] = None) -> str:
        """Get database URL with optional database name override.

        Args:
            dbname: Optional database name to override default

        Returns:
            Complete database URL string
        """
        db = dbname or self.DB_NAME
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{db}"

    @property
    def DB_URL(self) -> str:
        """Get the default database URL.

        Returns:
            Complete database URL string
        """
        return self.get_db_url()

    @property
    def TEST_DB_URL(self) -> str:
        """Get the test database URL.

        Returns:
            Complete test database URL string
        """
        return self.get_db_url(self.TEST_DB_NAME)

    def as_dict(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary.

        Returns:
            Dictionary of all configuration values
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and isinstance(value, (str, int, float, list))
        }
