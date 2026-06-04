from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard.
    raise SystemExit(
        "PyYAML is required. Install the project environment from environment.yml."
    ) from exc


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


config: Optional[dict] = None


def load_config(path: Path = 'datasets.yaml') -> dict[str, Any]:
    global config
    if config is None:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return config
