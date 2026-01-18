"""Firecrawl deep web crawler integration."""

import asyncio
from typing import Any

from .llm import LLMRouter


class FirecrawlClient:
    """Client for Firecrawl deep web crawling."""

    def __init__(
        self,
        llm_router: LLMRouter,
        base_url: str = "http://localhost:3002",
        api_key: str | None = None,
        max_concurrency: int = 4,
        timeout: int = 120,
    ):
        """Initialize Firecrawl client.

        Args:
            llm_router: LLM routing service
            base_url: Firecrawl API base URL
            api_key: API key
            max_concurrency: Maximum concurrent requests
            timeout: Request timeout
        """
        self.llm_router = llm_router
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrency = max_concurrency
        self.timeout = timeout

    async def scrape_url(
        self,
        url: str,
        extract_images: bool = False,
        extract_tables: bool = False,
    ) -> dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            extract_images: Extract images
            extract_tables: Extract tables

        Returns:
            Scraped content with metadata
        """
        if not self.api_key:
            return {"error": "API key not configured"}

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "url": url,
            "extractImages": extract_images,
            "extractTables": extract_tables,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/scrape",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def batch_scrape(
        self,
        urls: list[str],
        extract_images: bool = False,
        extract_tables: bool = False,
    ) -> list[dict[str, Any]]:
        """Scrape multiple URLs in batch.

        Args:
            urls: List of URLs to scrape
            extract_images: Extract images
            extract_tables: Extract tables

        Returns:
            List of scraped results
        """
        tasks = [
            self.scrape_url(
                url=url,
                extract_images=extract_images,
                extract_tables=extract_tables,
            )
            for url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]

    async def crawl_site(
        self,
        start_url: str,
        limit: int = 50,
        exclude_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl an entire website.

        Args:
            start_url: Starting URL for crawling
            limit: Maximum pages to crawl
            exclude_paths: Paths to exclude

        Returns:
            Crawling results
        """
        if not self.api_key:
            return {"error": "API key not configured"}

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "url": start_url,
            "limit": limit,
            "excludePaths": exclude_paths or [],
        }

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self.base_url}/v1/crawl",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def map_site(
        self,
        url: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Create site map of website.

        Args:
            url: URL to map
            limit: Maximum pages

        Returns:
            Site map with links
        """
        if not self.api_key:
            return {"error": "API key not configured"}

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "url": url,
            "limit": limit,
            "sitemap": True,
        }

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self.base_url}/v1/map",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """Check Firecrawl client health.

        Returns:
            Health status
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/v1/status")
                response.raise_for_status()
                return response.json()

        except Exception:
            return {
                "firecrawl": {"status": "error", "message": "Health check failed"},
                "overall": "degraded",
            }
