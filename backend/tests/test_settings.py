"""Tests for JARVIS settings, loopback validation, and asset discovery."""
import pytest
from pydantic import ValidationError

from config.settings import Settings, settings


def test_project_root_and_assets_dir_exist():
    assert settings.project_root.is_dir()
    assert settings.assets_dir.is_dir()
    assert (settings.project_root / "assets").resolve() == settings.assets_dir.resolve()


def test_asset_paths_and_files_exist():
    # Icon verification
    assert settings.icon_path.name == "JARVIS.jpg"
    assert settings.icon_path.is_file()
    assert settings.icon_path.stat().st_size > 1000

    # Music verification
    assert settings.music_path.name == "garage_music.mp3"
    assert settings.music_path.is_file()
    assert settings.music_path.stat().st_size > 100000

    # Clean voice prompt verification
    assert settings.voice_clean_path.name == "jarvis_voice_clean.wav"
    assert settings.voice_clean_path.is_file()
    assert settings.voice_clean_path.stat().st_size > 10000

    # Voice conditioning cache verification
    assert settings.voice_cond_path.name == ".jarvis_cond6.wav"
    assert settings.voice_cond_path.is_file()
    assert settings.voice_cond_path.stat().st_size > 10000


def test_asset_catalog_and_health_reporting():
    catalog = settings.get_asset_catalog()
    assert set(catalog.keys()) == {"icon", "music", "voice_clean", "voice_cond"}
    for asset_info in catalog.values():
        assert asset_info["exists"] is True
        assert asset_info["size_bytes"] > 0
        assert asset_info["status"] == "ready"

    assert settings.verify_all_assets_present() is True


def test_ollama_safe_defaults():
    assert settings.ollama_base_url.startswith("http://127.0.0.1") or settings.ollama_base_url.startswith("http://localhost")
    assert settings.llm_model == "qwen3:4b"
    assert settings.llm_num_ctx == 2048
    assert settings.vision_model == "qwen3.5:4b"


def test_ollama_loopback_validator():
    # Loopback URLs should be accepted
    s1 = Settings(OLLAMA_BASE_URL="http://127.0.0.1:11434")
    assert s1.ollama_base_url == "http://127.0.0.1:11434"

    s2 = Settings(OLLAMA_BASE_URL="http://localhost:11434/")
    assert s2.ollama_base_url == "http://localhost:11434"

    # Non-loopback URLs should raise validation error for safety
    with pytest.raises(ValidationError):
        Settings(OLLAMA_BASE_URL="http://remote-server.com:11434")

    with pytest.raises(ValidationError):
        Settings(OLLAMA_BASE_URL="https://external-api.com")
