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
