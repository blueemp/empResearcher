"""Rerank service for document relevance scoring."""

from typing import Any

from .llm import LLMRouter


class RerankerService:
    """Service for reranking documents by relevance."""

    def __init__(self, llm_router: LLMRouter):
        """Initialize reranker service.

        Args:
            llm_router: LLM routing service
        """
        self.llm_router = llm_router

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by query relevance.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            model_name: Specific rerank model (optional)

        Returns:
            Reranked documents with scores
        """
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        docs_text = [doc.get("content", "") for doc in documents]

        try:
            rerank_result = await self.llm_router.route_rerank(
                query=query,
                docs=docs_text,
                top_k=top_k,
            )

            reranked = []
            for doc_idx, score in rerank_result:
                if doc_idx < len(documents):
                    doc = documents[doc_idx].copy()
                    doc["rerank_score"] = score
                    reranked.append(doc)

            return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        except Exception as e:
            return documents[:top_k]

    async def multi_signal_rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank using multiple signals (relevance, trust, freshness).

        Args:
            query: Search query
            documents: List of documents to rerank
            weights: Signal weights (relevance, trust, freshness)

        Returns:
            Reranked documents with combined scores
        """
        if weights is None:
            weights = {
                "relevance": 0.5,
                "trust": 0.3,
                "freshness": 0.2,
            }

        for doc in documents:
            doc["final_score"] = (
                doc.get("relevance_score", 0.5) * weights["relevance"]
                + doc.get("trust_score", 0.5) * weights["trust"]
                + doc.get("freshness_score", 0.5) * weights["freshness"]
            )

        return sorted(documents, key=lambda x: x["final_score"], reverse=True)

    async def diversity_rerank(
        self,
        documents: list[dict[str, Any]],
        diversity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Rerank with diversity to reduce redundancy.

        Args:
            documents: List of documents to rerank
            diversity_threshold: Similarity threshold for deduplication

        Returns:
            Diversified document list
        """
        if not documents:
            return []

        selected = []
        for doc in sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True):
            is_diverse = True

            for selected_doc in selected:
                similarity = self._calculate_similarity(
                    doc.get("content", ""),
                    selected_doc.get("content", ""),
                )

                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(doc)

        return selected

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0-1
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    async def batch_rerank(
        self,
        queries: list[tuple[str, list[dict[str, Any]]]],
        top_k_per_query: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Rerank multiple query-document pairs in batch.

        Args:
            queries: List of (query, documents) tuples
            top_k_per_query: Results per query

        Returns:
            Dictionary mapping queries to reranked results
        """
        results = {}

        for query, documents in queries:
            reranked = await self.rerank(query, documents, top_k=top_k_per_query)
            results[query] = reranked

        return results
