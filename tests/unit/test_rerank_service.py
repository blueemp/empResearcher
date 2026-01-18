"""Tests for rerank service."""

import pytest

from emp_researcher.services import RerankerService, LLMRouter


@pytest.fixture
def llm_router():
    """Mock LLM router."""
    return None


@pytest.mark.unit
def test_multi_signal_rerank():
    """Test multi-signal reranking."""
    documents = [
        {"content": "test doc 1", "relevance_score": 0.8, "trust_score": 0.9},
        {"content": "test doc 2", "relevance_score": 0.6, "trust_score": 0.5},
    ]

    reranker = RerankerService(mock_llm_router)
    results = reranker.multi_signal_rerank("test query", documents)

    assert len(results) == 2
    assert results[0]["final_score"] > results[1]["final_score"]


@pytest.mark.unit
def test_diversity_rerank():
    """Test diversity-based reranking."""
    documents = [
        {"content": "test doc 1 content about topic", "rerank_score": 0.9},
        {"content": "test doc 1 content about topic", "rerank_score": 0.85},
        {"content": "different topic entirely here", "rerank_score": 0.8},
    ]

    reranker = RerankerService(mock_llm_router)
    results = reranker.diversity_rerank(documents, diversity_threshold=0.7)

    assert len(results) <= 3


@pytest.mark.unit
def test_calculate_similarity():
    """Test similarity calculation."""
    reranker = RerankerService(mock_llm_router)

    similarity = reranker._calculate_similarity(
        "hello world test",
        "hello world",
    )

    assert 0 < similarity <= 1


def mock_llm_router() -> LLMRouter:
    """Mock LLM router for testing."""
    return None
