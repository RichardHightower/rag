"""Test OpenAI embedder."""

import os
from unittest.mock import Mock, patch

import pytest

from rag.config import EMBEDDING_DIM, OPENAI_MODEL
from rag.db import Chunk
from rag.embeddings.openai_embedder import OpenAIEmbedder


def test_init_with_api_key():
    """Test initialization with API key."""
    embedder = OpenAIEmbedder(api_key="test_key")
    assert embedder.model_name == OPENAI_MODEL
    assert embedder.dimension == EMBEDDING_DIM
    assert embedder.batch_size == 16


def test_init_without_api_key():
    """Test initialization without API key."""
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAIEmbedder(api_key="")


def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    embedder = OpenAIEmbedder(
        model_name="custom-model", dimension=128, api_key="test_key", batch_size=32
    )
    assert embedder.model_name == "custom-model"
    assert embedder.dimension == 128
    assert embedder.batch_size == 32


def test_get_dimension():
    """Test get_dimension method."""
    embedder = OpenAIEmbedder(api_key="test_key", dimension=256)
    assert embedder.get_dimension() == 256


@patch("openai.OpenAI")
def test_embed_texts_single_batch(mock_openai):
    """Test embedding texts that fit in a single batch."""
    # Mock response from OpenAI
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client = Mock()
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    embedder = OpenAIEmbedder(api_key="test_key", dimension=3, batch_size=5)
    texts = ["Hello", "World"]
    chunks = [Chunk(content=text) for text in texts]
    embeddings = embedder.embed_texts(chunks)

    # Verify the results
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify OpenAI was called correctly
    mock_client.embeddings.create.assert_called_once_with(
        model=OPENAI_MODEL, input=texts
    )


@patch("openai.OpenAI")
def test_embed_texts_multiple_batches(mock_openai):
    """Test embedding texts that require multiple batches."""
    # Mock responses for two batches
    mock_response1 = Mock()
    mock_response1.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
    mock_response2 = Mock()
    mock_response2.data = [Mock(embedding=[0.5, 0.6])]

    mock_client = Mock()
    mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]
    mock_openai.return_value = mock_client

    embedder = OpenAIEmbedder(api_key="test_key", dimension=2, batch_size=2)
    texts = ["Text1", "Text2", "Text3"]
    embeddings = embedder.embed_texts([Chunk(content=text) for text in texts])

    # Verify the results
    assert len(embeddings) == 3
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]
    assert embeddings[2] == [0.5, 0.6]

    # Verify OpenAI was called correctly for each batch
    assert mock_client.embeddings.create.call_count == 2
    mock_client.embeddings.create.assert_any_call(
        model=OPENAI_MODEL, input=["Text1", "Text2"]
    )
    mock_client.embeddings.create.assert_any_call(model=OPENAI_MODEL, input=["Text3"])
