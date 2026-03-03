"""Configuration loader — reads YAML or JSON config files.

Supports:
  - YAML and JSON config files
  - Environment variable overrides
  - Nested key access with dot notation
  - Default values
  - Config validation

The config hierarchy (later overrides earlier):
  1. Built-in defaults
  2. Config file (configs/default.yaml)
  3. Environment-specific file (configs/{ENV}.yaml)
  4. Environment variables (ETHICAGENT_*)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False


_DEFAULTS: dict[str, Any] = {
    "pipeline": {
        "dry_run": False,
        "max_retries": 3,
        "timeout_seconds": 30,
        "enable_caching": True,
    },
    "thresholds": {
        "approval": 0.80,
        "escalation": 0.50,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 1024,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    },
    "domains": {
        "default": "general",
        "supported": ["healthcare", "finance", "hiring", "disaster", "general"],
    },
}


class Config:
    """Application configuration manager.

    Usage::

        config = Config.from_file("configs/default.yaml")
        model = config.get("llm.model", "gpt-4")
        config.set("pipeline.dry_run", True)
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = _deep_merge(_DEFAULTS, data or {})
        self._apply_env_overrides()

    @classmethod
    def from_file(cls, filepath: str | Path) -> Config:
        """Load config from a YAML or JSON file."""
        path = Path(filepath)
        if not path.exists():
            logger.warning("Config file not found: %s, using defaults", path)
            return cls()

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required to load YAML config files")
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

        logger.info("Loaded config from %s", path)
        return cls(data)

    @classmethod
    def from_env(cls) -> Config:
        """Create config from environment variables only."""
        return cls()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation (e.g. 'llm.model')."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot notation."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def _apply_env_overrides(self) -> None:
        """Override config values from ETHICAGENT_* environment variables."""
        prefix = "ETHICAGENT_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("__", ".")
                # Try to parse as JSON for complex values
                try:
                    parsed = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed = value
                self.set(config_key, parsed)
                logger.debug("Env override: %s = %s", config_key, parsed)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Backwards-compatible alias used by orchestrator and other modules
ConfigLoader = Config
