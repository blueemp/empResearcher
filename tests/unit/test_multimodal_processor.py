"""Tests for multimodal processor."""

import pytest

from emp_researcher.services import MultimodalProcessor


@pytest.fixture
def multimodal_processor():
    """Mock multimodal processor."""
    return MultimodalProcessor(None)


@pytest.mark.unit
def test_process_image(multimodal_processor):
    """Test image processing."""
    result = multimodal_processor.process_image.__wrapped__(
        multimodal_processor, b"fake_image_data", "png"
    )
    assert result is not None
    assert result["modality"] == "image"


@pytest.mark.unit
def test_process_table(multimodal_processor):
    """Test table processing."""
    result = multimodal_processor.process_table.__wrapped__(
        multimodal_processor, "header1,header2\nrow1,row2", "csv"
    )
    assert result is not None
    assert result["modality"] == "table"


@pytest.mark.unit
def test_process_audio(multimodal_processor):
    """Test audio processing."""
    result = multimodal_processor.process_audio.__wrapped__(
        multimodal_processor, b"fake_audio_data", "mp3", 120.0
    )
    assert result is not None
    assert result["modality"] == "audio"


@pytest.mark.unit
def test_health_check(multimodal_processor):
    """Test health check."""
    health = multimodal_processor.health_check.__wrapped__(multimodal_processor)
    assert health["overall"] == "healthy"
