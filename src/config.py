"""Loads config.yaml so the rest of src/ can read project settings from one place."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml and return it as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
