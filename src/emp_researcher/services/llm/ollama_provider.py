"""Ollama LLM provider implementation."""

from typing import Any

import httpx

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)
        self.timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send chat completion request to Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Generate embeddings using Ollama.

        Note: Ollama requires embedding-capable models.
        """
        payload = {
            "model": model,
            "input": texts,
            **kwargs,
        }

        response = await self.client.post(
            f"{self.base_url}/api/embed",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def rerank(
        self,
        query: str,
        docs: list[str],
        model: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Rerank documents using embeddings."""
        import numpy as np

        if top_k is None:
            top_k = len(docs)

        query_result = await self.embed([query], model=model)
        doc_result = await self.embed(docs, model=model)

        query_vec = np.array(query_result["embeddings"][0])
        doc_vecs = np.array(doc_result["embeddings"])

        similarities = np.dot(doc_vecs, query_vec).tolist()
        indexed_scores = [(i, score) for i, score in enumerate(similarities)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    async def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return True
        except Exception:
            return False
