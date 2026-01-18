"""Web Searcher Agent for web information retrieval."""

import asyncio
from typing import Any

from ..services import LLMRouter, SearXNGClient


class WebSearcherAgent:
    """Agent for web search operations."""

    def __init__(
        self,
        llm_router: LLMRouter,
        searxng_client: SearXNGClient,
    ):
        """Initialize web searcher agent.

        Args:
            llm_router: LLM routing service
            searxng_client: SearXNG client
        """
        self.llm_router = llm_router
        self.searxng_client = searxng_client

    async def search_web(
        self,
        queries: list[str],
        engines_zh: list[str] | None = None,
        engines_en: list[str] | None = None,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Perform web search for multiple queries.

        Args:
            queries: List of search queries
            engines_zh: Chinese engines
            engines_en: English engines
            max_results: Results per query

        Returns:
            Combined search results
        """
        all_results = []

        for query in queries:
            if any("\u4e00-\u9fff" in c for c in query):
                results = await self.searxng_client.search(
                    query,
                    engines=engines_zh,
                    language="zh",
                )
            else:
                results = await self.searxng_client.search(
                    query,
                    engines=engines_en,
                    language="en",
                )

            all_results.extend(results)

        return all_results

    async def search_bilingual_parallel(
        self,
        query: str,
        max_results: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        """Perform parallel bilingual search.

        Args:
            query: Original query
            max_results: Maximum results per language

        Returns:
            Dictionary with zh and en results
        """
        translated_query = await self._translate_for_bilingual(query)

        queries_zh = [query] if any("\u4e00-\u9fff" in c for c in query) else [translated_query]
        queries_en = [query] if not any("\u4e00-\u9fff" in c for c in query) else [translated_query]

        results_zh = await self._search_with_engines(
            queries_zh, engines_zh=["baidu", "so", "google"]
        )
        results_en = await self._search_with_engines(
            queries_en, engines_en=["google", "bing", "duckduckgo"]
        )

        return {
            "zh": results_zh[:max_results],
            "en": results_en[:max_results],
            "combined": results_zh[:max_results] + results_en[:max_results],
        }

    async def _translate_for_bilingual(self, query: str) -> str:
        """Translate query for bilingual search.

        Args:
            query: Original query

        Returns:
            Translated query
        """
        is_chinese = any("\u4e00-\u9fff" in c for c in query)

        if is_chinese:
            return await self._translate(query, "en")
        else:
            return await self._translate(query, "zh")

    async def _translate(
        self,
        text: str,
        target_lang: str,
    ) -> str:
        """Translate text using LLM.

        Args:
            text: Text to translate
            target_lang: Target language (zh/en)

        Returns:
            Translated text
        """
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator for research purposes.",
            },
            {
                "role": "user",
                "content": f"Translate to {target_lang}: {text}",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="bilingual_translation",
        )

        return response.choices[0].message.content

    async def _search_with_engines(
        self,
        queries: list[str],
        engines: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search with specified engines.

        Args:
            queries: List of search queries
            engines: Specific engines (optional)

        Returns:
            Search results
        """
        results = []

        for query in queries:
            search_result = await self.searxng_client.search(
                query,
                engines=engines,
            )
            results.extend(search_result)

        return results

    async def extract_and_rerank(
        self,
        urls: list[str],
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs and rerank by relevance.

        Args:
            urls: List of URLs to process
            query: Search query for reranking
            top_k: Number of top results to return

        Returns:
            Reranked documents
        """
        extracted = []

        for url in urls[:50]:
            content = await self._fetch_content(url)
            if content:
                extracted.append(
                    {
                        "url": url,
                        "content": content[:2000],
                        "extracted_at": "now",
                    }
                )

        if extracted:
            reranked = await self._rerank_documents(query, extracted, top_k)
            return reranked

        return []

    async def _fetch_content(self, url: str) -> str | None:
        """Fetch content from URL.

        Args:
            url: URL to fetch

        Returns:
            Extracted content or None
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    return response.text[:5000]
                return None
        except Exception:
            return None

    async def _rerank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank documents by query relevance.

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of top results

        Returns:
            Reranked documents
        """
        docs_text = [d.get("content", "") for d in documents]

        reranked = await self.llm_router.route_rerank(
            query=query,
            docs=docs_text,
            top_k=top_k,
        )

        results = []
        for doc_idx, score in reranked:
            if doc_idx < len(documents):
                doc = documents[doc_idx]
                doc["relevance_score"] = score
                results.append(doc)

        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]

    async def health_check(self) -> dict[str, Any]:
        """Check web searcher health.

        Returns:
            Health status
        """
        searxng_healthy = await self.searxng_client.health_check()

        return {
            "searxng": {"status": "ok" if searxng_healthy else "error"},
            "overall": "healthy" if searxng_healthy else "degraded",
        }
