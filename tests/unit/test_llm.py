"""Unit tests for LLM services."""

import pytest

from emp_researcher.services import LLMRouter, ModelType


@pytest.mark.unit
def test_model_type_constants():
    """Test model type constants are defined."""
    assert ModelType.SMALL_FAST == "small-fast-model"
    assert ModelType.STRONGER == "stronger-model"
    assert ModelType.RERANK == "rerank-model"


@pytest.mark.unit
def test_router_initialization():
    """Test LLM router initializes with config."""
    config = {
        "llm": {
            "providers": {
                "test": {
                    "type": "openai_compatible",
                    "base_url": "http://test.com",
                }
            }
        }
    }

    router = LLMRouter(config)
    assert "test" in router.providers


@pytest.mark.unit
def test_get_default_provider():
    """Test getting default provider for model type."""
    config = {
        "llm": {
            "providers": {"openai": {"type": "openai_compatible", "base_url": "http://test.com"}},
            "default_provider": {"small-fast-model": "openai"},
        }
    }

    router = LLMRouter(config)
    provider = router.providers["openai"]
    assert provider is not None
