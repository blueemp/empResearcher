"""Unit tests for configuration management."""

import os
import pytest
from pathlib import Path

from emp_researcher.utils import ConfigManager


@pytest.mark.unit
def test_config_manager_initializes():
    """Test config manager initializes."""
    manager = ConfigManager()
    assert manager.config is not None


@pytest.mark.unit
def test_get_config_value():
    """Test getting configuration value."""
    manager = ConfigManager()
    app_name = manager.get("app.name")
    assert app_name is not None or app_name == "emp-researcher"


@pytest.mark.unit
def test_expand_env_vars():
    """Test environment variable expansion."""
    os.environ["TEST_VAR"] = "test_value"

    config = {"key": "${TEST_VAR}"}
    manager = ConfigManager()
    result = manager._expand_env_vars(config)

    assert result == "test_value"


@pytest.mark.unit
def test_get_llm_config():
    """Test getting LLM configuration."""
    manager = ConfigManager()
    llm_config = manager.get_llm_config()
    assert isinstance(llm_config, dict)


@pytest.mark.unit
def test_get_search_config():
    """Test getting search configuration."""
    manager = ConfigManager()
    search_config = manager.get_search_config()
    assert isinstance(search_config, dict)
