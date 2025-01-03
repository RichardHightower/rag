"""Test text chunking utilities."""

import pytest
from rag.db.chunking import chunk_text


def test_basic_chunking():
    """Test basic text chunking with default parameters."""
    # Create test text with 10 lines
    text = '\n'.join([f'Line {i}' for i in range(10)])
    
    # Default chunk_size=500, overlap=50 should return single chunk
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_custom_chunk_size():
    """Test chunking with custom chunk size."""
    # Create test text with 10 lines
    lines = [f'Line {i}' for i in range(10)]
    text = '\n'.join(lines)
    
    # Set chunk_size to 4 lines, overlap to 1
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    
    # Expected chunks with overlap:
    # Chunk 1: lines 0-3
    # Chunk 2: lines 3-6
    # Chunk 3: lines 6-9
    assert len(chunks) == 3
    
    # Verify first chunk
    assert chunks[0] == '\n'.join(['Line 0', 'Line 1', 'Line 2', 'Line 3'])
    
    # Verify middle chunk has overlap
    assert chunks[1] == '\n'.join(['Line 3', 'Line 4', 'Line 5', 'Line 6'])
    
    # Verify last chunk
    assert chunks[2] == '\n'.join(['Line 6', 'Line 7', 'Line 8', 'Line 9'])


def test_empty_text():
    """Test chunking empty text."""
    chunks = chunk_text('')
    assert len(chunks) == 1
    assert chunks[0] == ''


def test_whitespace_text():
    """Test chunking whitespace text."""
    chunks = chunk_text('   \n  \n  ')
    assert len(chunks) == 1
    assert chunks[0] == '   \n  \n  '


def test_single_line():
    """Test chunking single line of text."""
    text = 'Single line'
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_smaller_than_chunk():
    """Test when text is smaller than chunk size."""
    text = '\n'.join([f'Line {i}' for i in range(5)])
    chunks = chunk_text(text, chunk_size=10, overlap=5)  # Make sure overlap < chunk_size
    assert len(chunks) == 1
    assert chunks[0] == text


def test_no_overlap():
    """Test chunking with no overlap."""
    # Create test text with 6 lines
    lines = [f'Line {i}' for i in range(6)]
    text = '\n'.join(lines)
    
    # Set chunk_size to 2 lines, no overlap
    chunks = chunk_text(text, chunk_size=2, overlap=0)
    
    # Expected chunks:
    # Chunk 1: lines 0-1
    # Chunk 2: lines 2-3
    # Chunk 3: lines 4-5
    assert len(chunks) == 3
    assert chunks[0] == '\n'.join(['Line 0', 'Line 1'])
    assert chunks[1] == '\n'.join(['Line 2', 'Line 3'])
    assert chunks[2] == '\n'.join(['Line 4', 'Line 5'])


def test_invalid_chunk_size():
    """Test invalid chunk size."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0)
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=-1)


def test_invalid_overlap():
    """Test invalid overlap size."""
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_text("test", chunk_size=2, overlap=-1)
    
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("test", chunk_size=2, overlap=2)
    
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("test", chunk_size=2, overlap=3)
