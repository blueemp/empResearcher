"""LLM routing service for provider selection and model routing."""

import asyncio
from typing import Any

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAICompatibleProvider


class ModelType:
    """Model type constants."""

    SMALL_FAST = "small-fast-model"
    STRONGER = "stronger-model"
    RERANK = "rerank-model"


class LLMRouter:
    """Routes LLM requests to appropriate provider and model."""

    def __init__(self, config: dict[str, Any]):
        """Initialize LLM router with configuration.

        Args:
            config: Configuration dictionary with providers and routing rules
        """
        self.config = config
        self.providers: dict[str, LLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_configs = self.config.get("llm", {}).get("providers", {})

        for name, pconfig in provider_configs.items():
            provider_type = pconfig.get("type")

            if provider_type == "ollama":
                self.providers[name] = OllamaProvider(
                    base_url=pconfig.get("base_url", "http://localhost:11434"),
                    timeout=pconfig.get("timeout", 120),
                )
            else:
                self.providers[name] = OpenAICompatibleProvider(
                    base_url=pconfig["base_url"],
                    api_key=pconfig.get("api_key"),
                    timeout=pconfig.get("timeout", 60),
                )

    async def get_provider(
        self,
        provider_name: str | None = None,
        model_type: str | None = None,
    ) -> LLMProvider:
        """Get a provider for the given task.

        Args:
            provider_name: Specific provider name (optional)
            model_type: Model type for routing (optional)

        Returns:
            LLM provider instance

        Raises:
            ValueError: If no provider found
        """
        if provider_name:
            return self.providers[provider_name]

        if model_type:
            default_provider = (
                self.config.get("llm", {}).get("default_provider", {}).get(model_type)
            )

            if default_provider:
                provider = self.providers.get(default_provider)
                if provider:
                    return provider

        return next(iter(self.providers.values()))

    async def route_chat(
        self,
        messages: list[dict[str, Any]],
        task_type: str | None = None,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route chat request to appropriate provider and model.

        Args:
            messages: Chat messages
            task_type: Task type for model selection (optional)
            provider_name: Force specific provider (optional)
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model_type = self._get_model_type_for_task(task_type)
        provider = await self.get_provider(provider_name, model_type)

        model = self._get_model_for_task(provider_name, model_type)

        return await provider.chat(messages, model=model, **kwargs)

    async def route_embed(
        self,
        texts: list[str],
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route embedding request.

        Args:
            texts: Texts to embed
            provider_name: Force specific provider (optional)
            **kwargs: Additional parameters

        Returns:
            Embedding response
        """
        provider = await self.get_provider(provider_name, None)
        model = self._get_model_for_task(provider_name, None)
        return await provider.embed(texts, model=model, **kwargs)

    async def route_rerank(
        self,
        query: str,
        docs: list[str],
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Route rerank request.

        Args:
            query: Query string
            docs: Documents to rerank
            provider_name: Force specific provider (optional)
            **kwargs: Additional parameters

        Returns:
            List of (doc_index, score) tuples
        """
        provider = await self.get_provider(provider_name, ModelType.RERANK)
        model = self._get_model_for_task(provider_name, ModelType.RERANK)
        return await provider.rerank(query, docs, model=model, **kwargs)

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all providers.

        Returns:
            Dict mapping provider names to health status
        """
        results = {}
        tasks = [(name, provider.health_check()) for name, provider in self.providers.items()]

        for name, task in tasks:
            results[name] = await task

        return results

    def _get_model_type_for_task(self, task_type: str | None) -> str | None:
        """Get model type for a task."""
        if not task_type:
            return None

        mapping = self.config.get("llm", {}).get("task_model_mapping", {})
        return mapping.get(task_type)

    def _get_model_for_task(
        self,
        provider_name: str | None,
        model_type: str | None,
    ) -> str:
        """Get specific model name for provider and model type."""
        if not provider_name or not model_type:
            provider_name, model_type = self._get_defaults()

        provider_models = self.config.get("llm", {}).get("provider_models", {})
        provider_config = provider_models.get(provider_name, {})

        model_list = provider_config.get(f"{model_type}s", [])
        if model_list:
            return model_list[0]

        return "gpt-3.5-turbo"

    def _get_defaults(self) -> tuple[str, str]:
        """Get default provider and model type."""
        defaults = self.config.get("llm", {}).get("default_provider", {})
        provider = defaults.get(ModelType.SMALL_FAST) or next(iter(self.providers.keys()))
        return provider, ModelType.SMALL_FAST
