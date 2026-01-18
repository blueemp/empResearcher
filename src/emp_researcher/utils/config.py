"""Configuration management module."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration."""

    app: dict[str, Any]
    vector_store: dict[str, Any]
    graph_store: dict[str, Any]
    database: dict[str, Any]
    documents: dict[str, Any]
    logging: dict[str, Any]
    observability: dict[str, Any]
    tasks: dict[str, Any]
    research: dict[str, Any]


class ConfigManager:
    """Manages application configuration from YAML files."""

    def __init__(self, config_dir: str | None = None):
        """Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = self._find_config_dir()

        self.config_dir = Path(config_dir)
        self.config: dict[str, Any] = {}
        self._load_configs()

    def _find_config_dir(self) -> str:
        """Find configuration directory."""
        cwd = Path.cwd()

        possible_paths = [
            cwd / "config",
            cwd.parent / "config",
            Path(__file__).parent.parent.parent / "config",
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)

        return "config"

    def _load_configs(self) -> None:
        """Load all YAML configuration files."""
        config_files = [
            "app_config.yaml",
            "llm_providers.yaml",
            "search_config.yaml",
        ]

        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config:
                        self._expand_env_vars(config)
                        self.config.update(config)

    def _expand_env_vars(self, config: Any) -> Any:
        """Expand environment variables in configuration.

        Args:
            config: Configuration value (dict, list, or primitive)

        Returns:
            Configuration with environment variables expanded
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}

        if isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]

        if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)

        return config

    def get(self, key: str, default: Any | None = None) -> Any:
        """Get configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., "app.name")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_llm_config(self) -> dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get("llm", {})

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration."""
        return self.config.get("search", {})

    def get_app_config(self) -> AppConfig:
        """Get application configuration as Pydantic model."""
        return AppConfig(**self.config)


_global_config: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get global configuration instance.

    Returns:
        ConfigManager instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigManager()

    return _global_config
