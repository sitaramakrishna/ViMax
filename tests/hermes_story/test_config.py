import pytest

from hermes_story import config


def test_example_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    c = config.load()  # falls back to configs/hermes-story.example.yaml
    assert c.chat_model == "google/gemma-4-31b-it"
    assert c.video_model == "google/veo-3.1-lite" and c.video_duration_s == 8
    assert c.default_cap_usd == 10.0
    with pytest.raises(config.ConfigError):
        c.api_key()


def test_key_from_env_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    c = config.load()
    assert c.api_key() == "sk-test"
    assert "sk-test" not in c.model_dump_json()
