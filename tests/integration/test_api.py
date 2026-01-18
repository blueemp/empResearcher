"""Integration tests for FastAPI application."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    async with AsyncClient(app=None, base_url="http://test") as client:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_task():
    """Test creating a research task."""
    task_data = {
        "query": "Test query",
        "depth": "standard",
        "output_format": "markdown",
    }

    async with AsyncClient(app=None, base_url="http://test") as client:
        pass
