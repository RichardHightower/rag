#!/usr/bin/env python3
"""Database initialization script."""

import logging
from pathlib import Path

from sqlalchemy import create_engine, text

from rag.config import DB_URL, VECTOR_INDEX_LISTS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize database with all required schemas and indexes."""
    engine = create_engine(DB_URL)

    # Get the SQL scripts directory
    sql_dir = Path(__file__).parent.parent.parent / "db" / "sql" / "ddl"

    try:
        # Execute all SQL files in order
        for sql_file in sorted(sql_dir.glob("*.sql")):
            logger.info(f"Executing {sql_file.name}")

            # Read and parametrize the SQL
            sql = sql_file.read_text()
            sql = sql.replace(":vector_index_lists", str(VECTOR_INDEX_LISTS))

            # Execute the SQL script
            with engine.connect() as conn:
                # Split on semicolon to handle multiple statements
                statements = sql.split(";")
                for statement in statements:
                    if statement.strip():
                        try:
                            conn.execute(text(statement))
                            conn.commit()
                        except Exception as e:
                            logger.warning(f"Error executing statement: {e}")
                            # Continue with next statement as some might be conditional
                            continue

        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    init_database()
