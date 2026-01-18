"""SearXNG search client."""

import asyncio
from typing import Any

import httpx


class SearXNGClient:
    """Client for SearXNG meta-search API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/search",
        timeout: int = 30,
        max_results: int = 20,
    ):
        """Initialize SearXNG client.

        Args:
            base_url: SearXNG API base URL
            timeout: Request timeout in seconds
            max_results: Maximum results per query
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_results = max_results
        self.client = httpx.AsyncClient(timeout=timeout)

    async def search(
        self,
        query: str,
        engines: list[str] | None = None,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Perform search query.

        Args:
            query: Search query string
            engines: Specific engines to use (optional)
            language: Language filter (en, zh, etc.)

        Returns:
            List of search results
        """
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": language,
            "engines": ",".join(engines) if engines else None,
        }

        response = await self.client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])[: self.max_results]

        return [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
                "engine": result.get("engine"),
                "score": result.get("score", 0),
                "language": language,
            }
            for result in results
        ]

    async def search_parallel(
        self,
        queries: list[str],
        engines_zh: list[str] | None = None,
        engines_en: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perform parallel bilingual search.

        Args:
            queries: List of search queries
            engines_zh: Chinese engines
            engines_en: English engines

        Returns:
            Dictionary mapping queries to results
        """
        tasks = []

        for query in queries:
            if any("\u4e00-\u9fff" in c for c in query):
                tasks.append(self.search(query, engines=engines_zh, language="zh"))
            else:
                tasks.append(self.search(query, engines=engines_en, language="en"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        results_by_query = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                results_by_query[query] = []
            else:
                results_by_query[query] = result

        return results_by_query

    async def health_check(self) -> bool:
        """Check if SearXNG is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/config")
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
