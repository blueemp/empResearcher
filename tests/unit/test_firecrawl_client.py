"""Tests for Firecrawl client."""

import pytest

from emp_researcher.services import FirecrawlClient


@pytest.fixture
def firecrawl_client():
    """Mock Firecrawl client."""
    return FirecrawlClient(None, api_key="test_key")


@pytest.mark.unit
def test_scrape_url(firecrawl_client):
    """Test URL scraping."""
    result = firecrawl_client.scrape_url.__wrapped__(
        firecrawl_client, "https://example.com", False, False
    )
    assert result is not None


@pytest.mark.unit
def test_batch_scrape(firecrawl_client):
    """Test batch scraping."""
    urls = ["https://example.com/page1", "https://example.com/page2"]
    result = firecrawl_client.batch_scrape.__wrapped__(firecrawl_client, urls, False, False)
    assert len(result) == 2


@pytest.mark.unit
def test_crawl_site(firecrawl_client):
    """Test site crawling."""
    result = firecrawl_client.crawl_site.__wrapped__(firecrawl_client, "https://example.com", 10)
    assert result is not None


@pytest.mark.unit
def test_map_site(firecrawl_client):
    """Test site mapping."""
    result = firecrawl_client.map_site.__wrapped__(firecrawl_client, "https://example.com", 50)
    assert result is not None


@pytest.mark.unit
def test_health_check(firecrawl_client):
    """Test health check."""
    health = firecrawl_client.health_check.__wrapped__(firecrawl_client)
    assert health is not None
