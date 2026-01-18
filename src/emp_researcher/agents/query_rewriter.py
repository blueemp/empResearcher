"""Query Rewriter Agent for intent understanding and query expansion."""

from typing import Any

from ..services import LLMRouter


class QueryRewriterAgent:
    """Agent for query rewriting and intent understanding."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize query rewriter agent.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def rewrite_query(
        self,
        query: str,
        language: str | None = None,
        depth: str = "standard",
    ) -> dict[str, Any]:
        """Rewrite and expand query for better search.

        Args:
            query: Original user query
            language: Source language (zh/en/mixed)
            depth: Research depth level

        Returns:
            Dictionary with rewritten queries and keywords
        """
        messages = [
            {
                "role": "system",
                "content": "You are a query rewriting specialist. Analyze the user's research query and generate optimized search queries.",
            },
            {
                "role": "user",
                "content": f"""Original query: {query}
Language: {language or "auto"}
Research depth: {depth}

Please analyze and provide:
1. Intent understanding: What is the user really asking for?
2. Sub-queries: 3-5 specific sub-queries to explore different aspects
3. Keywords: Extract key terms and synonyms
4. Search strategy: Suggested search engines and filters

Return JSON with keys: intent, sub_queries (list), keywords (list), strategy (dict).""",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="query_rewrite",
        )

        try:
            import json

            result = json.loads(response.choices[0].message.content)
            return {
                "original_query": query,
                "intent": result.get("intent", ""),
                "sub_queries": result.get("sub_queries", []),
                "keywords": result.get("keywords", []),
                "strategy": result.get("strategy", {}),
            }
        except Exception:
            return {
                "original_query": query,
                "intent": "General research",
                "sub_queries": [query],
                "keywords": query.split(),
                "strategy": {},
            }

    async def translate_query(
        self,
        query: str,
        target_lang: str = "en",
    ) -> str:
        """Translate query for bilingual search.

        Args:
            query: Original query
            target_lang: Target language (zh/en)

        Returns:
            Translated query
        """
        messages = [
            {
                "role": "system",
                "content": f"You are a professional translator. Translate queries accurately for research purposes.",
            },
            {
                "role": "user",
                "content": f"Translate to {target_lang}: {query}",
            },
        ]

        response = await self.llm_router.route_chat(
            messages=messages,
            task_type="bilingual_translation",
        )

        return response.choices[0].message.content

    async def generate_search_plan(
        self,
        rewritten_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate search plan from rewritten query.

        Args:
            rewritten_result: Result from rewrite_query

        Returns:
            List of search tasks
        """
        search_plan = []

        for sub_query in rewritten_result["sub_queries"]:
            search_plan.append(
                {
                    "query": sub_query,
                    "type": "web",
                    "priority": len(search_plan),
                    "status": "pending",
                }
            )

        return search_plan
