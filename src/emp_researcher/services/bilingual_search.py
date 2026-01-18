"""Bilingual search service for parallel zh/en queries."""

import asyncio
from typing import Any

from ..services import LLMRouter, QueryRewriterAgent, SearXNGClient


class BilingualSearchService:
    """Service for parallel Chinese/English search and result fusion."""

    def __init__(
        self,
        llm_router: LLMRouter,
        query_rewriter: QueryRewriterAgent,
        searxng_client: SearXNGClient,
        language_balance: float = 0.5,
        max_results_per_lang: int = 30,
    ):
        """Initialize bilingual search service.

        Args:
            llm_router: LLM routing service
            query_rewriter: Query rewriter agent
            searxng_client: SearXNG client
            language_balance: Language balance weight (0-1)
            max_results_per_lang: Max results per language
        """
        self.llm_router = llm_router
        self.query_rewriter = query_rewriter
        self.searxng_client = searxng_client
        self.language_balance = language_balance
        self.max_results_per_lang = max_results_per_lang

    async def search_bilingual(
        self,
        query: str,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Perform parallel bilingual search.

        Args:
            query: Original query
            max_iterations: Maximum search iterations

        Returns:
            Fused bilingual search results
        """
        detected_lang = self._detect_language(query)

        zh_query = await self._prepare_zh_query(query, detected_lang)
        en_query = await self._prepare_en_query(query, detected_lang)

        all_results = []

        for iteration in range(max_iterations):
            zh_results = await self._search_zh(zh_query, iteration)
            en_results = await self._search_en(en_query, iteration)

            zh_relevance = [r.get("relevance_score", 0) for r in zh_results]
            en_relevance = [r.get("relevance_score", 0) for r in en_results]

            if max(zh_relevance, default=0) < 0.3 and max(en_relevance, default=0) > 0.7:
                iteration_results = await self._expand_en_search(en_query, iteration)
            elif max(en_relevance, default=0) < 0.3 and max(zh_relevance, default=0) > 0.7:
                iteration_results = await self._expand_zh_search(zh_query, iteration)
            else:
                iteration_results = await self._merge_results(zh_results[:10], en_results[:10])

            all_results.extend(iteration_results)

        fused_results = await self._fuse_results(all_results)

        return {
            "query": query,
            "detected_language": detected_lang,
            "zh_query": zh_query,
            "en_query": en_query,
            "total_results": len(fused_results),
            "zh_count": sum(1 for r in fused_results if r.get("language") == "zh"),
            "en_count": sum(1 for r in fused_results if r.get("language") == "en"),
            "results": fused_results,
        }

    def _detect_language(self, text: str) -> str:
        """Detect language of input text.

        Args:
            text: Input text

        Returns:
            Language code (zh/en/mixed)
        """
        zh_chars = any("\u4e00-\u9fff" in c for c in text)

        if zh_chars:
            return "zh" if zh_chars / len(text) > 0.3 else "mixed"
        else:
            return "en"

    async def _prepare_zh_query(
        self,
        original_query: str,
        detected_lang: str,
    ) -> str:
        """Prepare Chinese query for search.

        Args:
            original_query: Original query
            detected_lang: Detected language

        Returns:
            Prepared Chinese query
        """
        if detected_lang == "zh":
            return original_query

        translated = await self.query_rewriter.translate_query(
            original_query,
            target_lang="zh",
        )

        messages = [
            {"role": "system", "content": "You are a query optimizer for Chinese search."},
            {"role": "user", "content": f"Optimize this query for Chinese search: {translated}"},
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="query_rewrite",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result.get("sub_queries", [original_query])[0]
        except Exception:
            return original_query

    async def _prepare_en_query(
        self,
        original_query: str,
        detected_lang: str,
    ) -> str:
        """Prepare English query for search.

        Args:
            original_query: Original query
            detected_lang: Detected language

        Returns:
            Prepared English query
        """
        if detected_lang == "en":
            return original_query

        translated = await self.query_rewriter.translate_query(
            original_query,
            target_lang="en",
        )

        messages = [
            {"role": "system", "content": "You are a query optimizer for English search."},
            {"role": "user", "content": f"Optimize this query for English search: {translated}"},
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="query_rewrite",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return result.get("sub_queries", [original_query])[0]
        except Exception:
            return original_query

    async def _search_zh(
        self,
        query: str,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Search Chinese sources.

        Args:
            query: Search query
            iteration: Search iteration number

        Returns:
            List of search results
        """
        try:
            engines_zh = ["baidu", "so", "google"]
            results = await self.searxng_client.search(
                query,
                engines=engines_zh,
                language="zh",
            )

            for result in results:
                result["language"] = "zh"
                result["iteration"] = iteration
                result["query_type"] = "zh"

            return results
        except Exception:
            return []

    async def _search_en(
        self,
        query: str,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Search English sources.

        Args:
            query: Search query
            iteration: Search iteration number

        Returns:
            List of search results
        """
        try:
            engines_en = ["google", "bing", "duckduckgo"]
            results = await self.searxng_client.search(
                query,
                engines=engines_en,
                language="en",
            )

            for result in results:
                result["language"] = "en"
                result["iteration"] = iteration
                result["query_type"] = "en"

            return results
        except Exception:
            return []

    async def _expand_zh_search(
        self,
        query: str,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Expand Chinese search with alternative queries.

        Args:
            query: Search query
            iteration: Search iteration number

        Returns:
            List of search results
        """
        messages = [
            {"role": "system", "content": "You are a search expansion specialist for Chinese."},
            {
                "role": "user",
                "content": f"Generate 3 alternative Chinese search queries for: {query}",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="query_rewrite",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            alt_queries = result.get("sub_queries", [query])

            all_results = []
            for alt_query in alt_queries[:3]:
                results = await self._search_zh(alt_query, iteration)
                all_results.extend(results)

            return all_results
        except Exception:
            return []

    async def _expand_en_search(
        self,
        query: str,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Expand English search with alternative queries.

        Args:
            query: Search query
            iteration: Search iteration number

        Returns:
            List of search results
        """
        messages = [
            {"role": "system", "content": "You are a search expansion specialist for English."},
            {
                "role": "user",
                "content": f"Generate 3 alternative English search queries for: {query}",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="query_rewrite",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            alt_queries = result.get("sub_queries", [query])

            all_results = []
            for alt_query in alt_queries[:3]:
                results = await self._search_en(alt_query, iteration)
                all_results.extend(results)

            return all_results
        except Exception:
            return []

    async def _merge_results(
        self,
        zh_results: list[dict[str, Any]],
        en_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge zh and en results with language balance.

        Args:
            zh_results: Chinese search results
            en_results: English search results

        Returns:
            Merged results
        """
        zh_weight = self.language_balance
        en_weight = 1.0 - self.language_balance

        merged = []

        for zh_result in zh_results:
            adjusted_score = zh_result.get("relevance_score", 0.5) * zh_weight
            result = zh_result.copy()
            result["adjusted_score"] = adjusted_score
            result["source_type"] = "web_zh"
            merged.append(result)

        for en_result in en_results:
            adjusted_score = en_result.get("relevance_score", 0.5) * en_weight
            result = en_result.copy()
            result["adjusted_score"] = adjusted_score
            result["source_type"] = "web_en"
            merged.append(result)

        return merged

    async def _fuse_results(
        self,
        all_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse and deduplicate all results.

        Args:
            all_results: All search results

        Returns:
            Deduplicated and fused results
        """
        seen_urls = set()
        fused = []

        for result in all_results:
            url = result.get("url", "")

            if url in seen_urls:
                continue

            if self._is_priority_source(result):
                result["priority_boost"] = 1.2
            else:
                result["priority_boost"] = 1.0

            result["final_score"] = (
                result.get("relevance_score", 0.5) * 0.4
                + result.get("priority_boost", 1.0) * 0.3
                + result.get("adjusted_score", 0.5) * 0.3
            )

            seen_urls.add(url)
            fused.append(result)

        return sorted(fused, key=lambda x: x["final_score"], reverse=True)[
            : self.max_results_per_lang * 2
        ]

    def _is_priority_source(self, result: dict[str, Any]) -> bool:
        """Check if result is from priority source.

        Args:
            result: Search result

        Returns:
            True if priority source
        """
        url = result.get("url", "")

        priority_domains = [
            "arxiv.org",
            "github.com",
            "acm.org",
            "cn",
            "sohu.com",
            "csdn.net",
            "zhihu.com",
        ]

        return any(domain in url for domain in priority_domains)

    async def health_check(self) -> dict[str, Any]:
        """Check bilingual search health.

        Returns:
            Health status
        """
        searxng_healthy = await self.searxng_client.health_check()

        return {
            "searxng": {"status": "ok" if searxng_healthy else "error"},
            "llm_router": await self.llm_router.health_check_all(),
            "query_rewriter": {"status": "ok"},
            "overall": "healthy" if searxng_healthy else "degraded",
        }
