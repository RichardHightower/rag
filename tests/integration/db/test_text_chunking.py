"""Integration tests for text chunking with database."""

import pytest
from sqlalchemy.orm import sessionmaker

from rag.chunking import split_text_into_chunks
from rag.db.dimension_utils import ensure_vector_dimension
from rag.db.models import Chunk


def test_chunk_database_integration(test_db):
    """Test that chunks can be stored and retrieved from database."""
    # Create test chunks
    text = "This is a test text for database integration"
    chunks = split_text_into_chunks(text, chunk_size=10, overlap=2)
    
    # Create database session
    Session = sessionmaker(bind=test_db)
    
    try:
        with Session() as session:
            # Ensure vector dimension is set to 1536
            ensure_vector_dimension(test_db, 1536)
            
            # Store chunks in database
            db_chunks = []
            for chunk in chunks:
                db_chunk = Chunk(
                    content=chunk.content,
                    start_char=chunk.start,
                    end_char=chunk.end,
                    embedding=[0.0] * 1536,  # Dummy embedding
                    chunk_index=len(db_chunks),
                    file_id=1  # Dummy file_id
                )
                db_chunks.append(db_chunk)
            
            session.add_all(db_chunks)
            session.commit()
            
            # Retrieve chunks
            stored_chunks = session.query(Chunk).order_by(Chunk.chunk_index).all()
            
            # Verify chunks were stored correctly
            assert len(stored_chunks) == len(chunks)
            for original, stored in zip(chunks, stored_chunks):
                assert stored.content == original.content
                assert stored.start_char == original.start
                assert stored.end_char == original.end
                
            # Verify chunk sequence reconstructs original text
            reconstructed = ""
            last_end = 0
            for stored in stored_chunks:
                if stored.start_char > last_end:
                    reconstructed += text[last_end:stored.start_char]
                reconstructed += stored.content
                last_end = stored.end_char
            
            if last_end < len(text):
                reconstructed += text[last_end:]
            
            assert reconstructed == text
    
    finally:
        # Clean up
        with Session() as session:
            session.query(Chunk).delete()
            session.commit()
