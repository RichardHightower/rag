"""Tests for text chunking functionality."""

import pytest
from rag.chunking import split_text_into_chunks


def test_empty_text():
    """Test chunking empty text."""
    chunks = split_text_into_chunks("")
    assert len(chunks) == 0


def test_text_smaller_than_chunk_size():
    """Test chunking text smaller than chunk size."""
    text = "Small text"
    chunks = split_text_into_chunks(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)


def test_text_larger_than_chunk_size():
    """Test chunking text larger than chunk size."""
    text = "This is a longer text that should be split into multiple chunks based on the specified chunk size."
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    # Check that chunks cover all text
    full_text = ""
    for chunk in chunks:
        full_text += chunk.text + " "
    assert text in full_text.strip()


def test_chunk_overlap():
    """Test that chunks properly overlap."""
    text = "This is a test of the chunking overlap functionality"
    chunk_size = 20
    overlap = 10
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    for i in range(len(chunks) - 1):
        # Get the overlapping text
        chunk1_end = chunks[i].text[-overlap:]
        chunk2_start = chunks[i + 1].text[:overlap]
        # There should be some common text between consecutive chunks
        assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0


def test_no_tiny_final_chunk():
    """Test that very small remaining text is merged with previous chunk."""
    text = "This is a test text that should not create a tiny final chunk"
    chunk_size = 20
    overlap = 5
    
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    # The last chunk should not be very small
    assert len(chunks[-1].text) > overlap
