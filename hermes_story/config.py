"""Configuration for hermes_story.

The OpenRouter key is deliberately not a config field: it is read from the
``OPENROUTER_API_KEY`` environment variable only.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"
EXAMPLE_CONFIG = CONFIG_DIR / "hermes-story.example.yaml"
LOCAL_CONFIG = CONFIG_DIR / "hermes-story.yaml"


class ConfigError(Exception):
    """Missing or malformed configuration."""


class StoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chat_model: str = "google/gemma-4-31b-it"
    image_model: str = "google/gemini-2.5-flash-image"
    video_model: str = "google/veo-3.1-lite"
    video_resolution: str = "720p"
    video_duration_s: int = 8
    num_image_candidates: int = 2
    sheet_image_model: str = "openai/gpt-image-2"
    sheet_image_quality: str = "medium"
    price_image_call: float = 0.04
    price_sheet_image_call: float = 0.06
    price_video_second: float = 0.03
    price_chat_call: float = 0.002
    default_cap_usd: float = 10.0
    base_url: str = "https://openrouter.ai/api/v1"

    def api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ConfigError("OPENROUTER_API_KEY is not set")
        return key


def load(path: str | None = None) -> StoryConfig:
    if path is None:
        path = os.environ.get("HERMES_STORY_CONFIG") or None
    if path is None:
        source = LOCAL_CONFIG if LOCAL_CONFIG.is_file() else EXAMPLE_CONFIG
    else:
        source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"config not found: {source}")
    except yaml.YAMLError as exc:
        raise ConfigError(f"config is not valid YAML: {source}: {exc}")
    if not isinstance(payload, dict):
        raise ConfigError(f"config must be a mapping: {source}")
    return StoryConfig(**payload)
