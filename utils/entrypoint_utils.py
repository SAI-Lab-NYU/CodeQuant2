from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.is_absolute() or config_path.exists():
        return config_path
    return REPO_ROOT / "config" / config_arg


def load_yaml_config(config_arg: str) -> dict:
    with open(resolve_config_path(config_arg), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None, "[ERROR] config is None."
    return config
