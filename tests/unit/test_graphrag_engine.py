"""Tests for GraphRAG engine."""

import pytest

from emp_researcher.services import GraphRAGEngine, LLMRouter


@pytest.fixture
def graph_rag_engine():
    """Mock GraphRAG engine."""
    return GraphRAGEngine(None)


@pytest.mark.unit
def test_extract_entities(graph_rag_engine):
    """Test entity extraction."""
    entities = graph_rag_engine.extract_entities.__wrapped__(
        graph_rag_engine, "test text", "doc123"
    )
    assert entities is not None


@pytest.mark.unit
def test_detect_communities(graph_rag_engine):
    """Test community detection."""
    communities = graph_rag_engine.detect_communities.__wrapped__(graph_rag_engine, 50)
    assert communities is not None


@pytest.mark.unit
def test_generate_community_summary(graph_rag_engine):
    """Test community summary generation."""
    summary = graph_rag_engine.generate_community_summary.__wrapped__(graph_rag_engine, "comm1", [])
    assert summary is not None


@pytest.mark.unit
def test_global_search(graph_rag_engine):
    """Test global search."""
    results = graph_rag_engine.global_search.__wrapped__(graph_rag_engine, "test query", 5)
    assert results is not None


@pytest.mark.unit
def test_local_search(graph_rag_engine):
    """Test local search."""
    results = graph_rag_engine.local_search.__wrapped__(graph_rag_engine, "test query", "comm1", 10)
    assert results is not None


@pytest.mark.unit
def test_health_check(graph_rag_engine):
    """Test health check."""
    health = graph_rag_engine.health_check.__wrapped__(graph_rag_engine)
    assert health["overall"] in ["healthy", "degraded"]
