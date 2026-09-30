"""Loading of the YAML files in config/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load(name: str, config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load config/<name>.yaml (``name`` without extension)."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    path = base / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
