"""LLM Provider abstraction layer."""

from abc import ABC, abstractmethod
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.create_embedding import CreateEmbeddingResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | list[ChatCompletionChunk]:
        """Send chat completion request to LLM.

        Args:
            messages: List of message dicts with role and content
            model: Model name to use
            stream: Whether to stream responses
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatCompletion object or list of chunks if streaming
        """
        pass

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings for texts.

        Args:
            texts: List of strings to embed
            model: Embedding model name
            **kwargs: Additional provider-specific parameters

        Returns:
            Embedding response with vectors
        """
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        docs: list[str],
        model: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Query string
            docs: List of document strings
            model: Rerank model name
            top_k: Number of top results to return
            **kwargs: Additional provider-specific parameters

        Returns:
            List of (doc_index, score) tuples sorted by relevance
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass
