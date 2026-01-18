"""OpenAI-compatible LLM provider implementation."""

import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API provider (OpenAI, SiliconFlow, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            base_url: API base URL
            api_key: API key (optional, reads from env if not provided)
            timeout: Request timeout in seconds
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            http_client=httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            ),
        )
        self.base_url = base_url

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send chat completion request."""
        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs,
        )
        return response

    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Generate embeddings for texts."""
        response = await self.client.embeddings.create(
            input=texts,
            model=model,
            **kwargs,
        )
        return response

    async def rerank(
        self,
        query: str,
        docs: list[str],
        model: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Rerank documents using cross-encoder model.

        Note: This is a basic implementation. For production,
        consider using specialized rerank APIs (e.g., BGE-Reranker).
        """
        import numpy as np

        if top_k is None:
            top_k = len(docs)

        query_embedding = await self.embed([query], model=model)
        doc_embeddings = await self.embed(docs, model=model)

        query_vec = np.array(query_embedding.data[0].embedding)
        doc_vecs = np.array([d.embedding for d in doc_embeddings.data])

        similarities = np.dot(doc_vecs, query_vec).tolist()
        indexed_scores = [(i, score) for i, score in enumerate(similarities)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
