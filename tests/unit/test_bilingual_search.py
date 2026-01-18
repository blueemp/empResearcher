"""Tests for bilingual search service."""

import pytest

from emp_researcher.services import BilingualSearchService


@pytest.fixture
def bilingual_service():
    """Mock bilingual search service."""
    return None


@pytest.mark.unit
def test_detect_language_zh():
    """Test Chinese language detection."""
    service = BilingualSearchService(None, None, None, language_balance=0.5)

    assert service._detect_language("这是一段中文文本") == "zh"


@pytest.mark.unit
def test_detect_language_en():
    """Test English language detection."""
    service = BilingualSearchService(None, None, None, language_balance=0.5)

    assert service._detect_language("This is English text") == "en"


@pytest.mark.unit
def test_detect_language_mixed():
    """Test mixed language detection."""
    service = BilingualSearchService(None, None, None, language_balance=0.5)

    assert service._detect_language("This English with 一些中文 mixed") == "mixed"


@pytest.mark.unit
def test_is_priority_source():
    """Test priority source detection."""
    service = BilingualSearchService(None, None, None, language_balance=0.5)

    assert service._is_priority_source({"url": "https://arxiv.org/abs/1234"}) is True
    assert service._is_priority_source({"url": "https://github.com/user/repo"}) is True
    assert service._is_priority_source({"url": "https://random-site.com/page"}) is False
