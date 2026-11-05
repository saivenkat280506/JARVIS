"""JARVIS Core Settings and Asset Discovery.

Provides validated local configuration and absolute paths to all media and model assets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory paths resolved relative to this file
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# Canonical asset file paths
ICON_PATH: Path = ASSETS_DIR / "icons" / "JARVIS.jpg"
MUSIC_PATH: Path = ASSETS_DIR / "music" / "garage_music.mp3"
VOICE_CLEAN_PATH: Path = ASSETS_DIR / "voices" / "jarvis_voice_clean.wav"
VOICE_COND_PATH: Path = ASSETS_DIR / "voices" / ".jarvis_cond6.wav"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATABASE_PATH: Path = DATA_DIR / "jarvis.db"
CHECKPOINT_DATABASE_PATH: Path = DATA_DIR / "checkpoints.db"



class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Project Paths
    project_root: Path = Field(default=PROJECT_ROOT)
    assets_dir: Path = Field(default=ASSETS_DIR)
    data_dir: Path = Field(
        default_factory=lambda: (
            Path(os.environ["JARVIS_DATA_DIR"]).resolve()
            if os.environ.get("JARVIS_DATA_DIR")
            else PROJECT_ROOT / "data"
        ),
        alias="JARVIS_DATA_DIR",
    )
    database_path: Path = Field(
        default_factory=lambda: (
            Path(os.environ["DATABASE_PATH"]).resolve()
            if os.environ.get("DATABASE_PATH")
            else (
                Path(os.environ["JARVIS_DATA_DIR"]).resolve() / "jarvis.db"
                if os.environ.get("JARVIS_DATA_DIR")
                else DATABASE_PATH
            )
        ),
        alias="DATABASE_PATH",
    )
    checkpoint_database_path: Path = Field(
        default_factory=lambda: (
            Path(os.environ["CHECKPOINT_DATABASE_PATH"]).resolve()
            if os.environ.get("CHECKPOINT_DATABASE_PATH")
            else (
                Path(os.environ["JARVIS_DATA_DIR"]).resolve() / "checkpoints.db"
                if os.environ.get("JARVIS_DATA_DIR")
                else CHECKPOINT_DATABASE_PATH
            )
        ),
        alias="CHECKPOINT_DATABASE_PATH",
    )
    enable_checkpoints: bool = Field(
        default=True,
        alias="ENABLE_CHECKPOINTS",
        description="Whether durable LangGraph checkpointing is active",
    )
    icon_path: Path = Field(default=ICON_PATH)
    music_path: Path = Field(default=MUSIC_PATH)
    voice_clean_path: Path = Field(default=VOICE_CLEAN_PATH)
    voice_cond_path: Path = Field(default=VOICE_COND_PATH)


    # Ollama Loopback Configuration
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
        description="Loopback URL for Ollama local inference daemon",
    )
    llm_model: str = Field(
        default="qwen3:4b",
        alias="LLM_MODEL",
        description="Primary local conversational LLM",
    )
    llm_num_ctx: int = Field(
        default=2048,
        alias="LLM_NUM_CTX",
        description="Bounded context window for low latency and low memory usage",
    )
    llm_temperature: float = Field(
        default=0.3,
        alias="LLM_TEMPERATURE",
        description="Inference sampling temperature",
    )
    llm_num_predict: int = Field(
        default=1024,
        alias="LLM_NUM_PREDICT",
        description="Bounded token generation cap per turn",
    )
    vision_model: str = Field(
        default="qwen3.5:4b",
        alias="VISION_MODEL",
        description="Multimodal local vision model",
    )

    # Conversation History & Memory
    conversation_max_turns: int = Field(
        default=5,
        alias="CONVERSATION_MAX_TURNS",
        description="Maximum complete conversational turns to include in context history",
    )
    conversation_max_chars: int = Field(
        default=6000,
        alias="CONVERSATION_MAX_CHARS",
        description="Maximum characters for restored conversation history to stay within Ollama context limit",
    )

    # Runtime Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Audio & Voice Settings
    sample_rate: int = Field(default=24000, alias="SAMPLE_RATE")
    stt_model_name: str = Field(default="base.en", alias="STT_MODEL_NAME")
    stt_device: str = Field(default="cpu", alias="STT_DEVICE")
    stt_compute_type: str = Field(default="int8", alias="STT_COMPUTE_TYPE")
    stt_sample_rate: int = Field(default=16000, alias="STT_SAMPLE_RATE")
    tts_sample_rate: int = Field(default=24000, alias="TTS_SAMPLE_RATE")
    stt_silence_duration_sec: float = Field(default=1.2, alias="STT_SILENCE_DURATION_SEC")
    stt_max_record_sec: float = Field(default=15.0, alias="STT_MAX_RECORD_SEC")
    voice_post_playback_cooldown_sec: float = Field(
        default=0.35,
        alias="VOICE_POST_PLAYBACK_COOLDOWN_SEC",
        description="Acoustic post-playback cooldown in seconds before microphone capture",
    )
    voice_reflection_similarity_threshold: float = Field(
        default=0.85,
        alias="VOICE_REFLECTION_SIMILARITY_THRESHOLD",
        description="Conservative similarity threshold for suppressing assistant speech reflections",
    )
    stt_calibration_duration_sec: float = Field(
        default=0.4,
        alias="STT_CALIBRATION_DURATION_SEC",
        description="Duration in seconds of initial audio chunks used to estimate ambient noise floor",
    )
    stt_noise_floor_multiplier: float = Field(
        default=2.0,
        alias="STT_NOISE_FLOOR_MULTIPLIER",
        description="Multiplier applied to measured noise floor to derive speech-start threshold",
    )
    stt_min_energy_threshold: float = Field(
        default=0.008,
        alias="STT_MIN_ENERGY_THRESHOLD",
        description="Minimum bound for speech-start RMS energy threshold",
    )
    stt_max_energy_threshold: float = Field(
        default=0.080,
        alias="STT_MAX_ENERGY_THRESHOLD",
        description="Maximum bound for speech-start RMS energy threshold",
    )
    stt_silence_threshold_multiplier: float = Field(
        default=1.3,
        alias="STT_SILENCE_THRESHOLD_MULTIPLIER",
        description="Multiplier applied to noise floor to determine silence endpointing threshold",
    )

    # Chrome DevTools Protocol (CDP) Settings
    cdp_host: str = Field(
        default="127.0.0.1",
        alias="CDP_HOST",
        description="Local loopback host for Chrome DevTools Protocol",
    )
    cdp_port: int = Field(
        default=9222,
        alias="CDP_PORT",
        description="Chrome remote debugging port",
    )
    cdp_timeout_sec: float = Field(
        default=5.0,
        alias="CDP_TIMEOUT_SEC",
        description="Timeout for CDP HTTP and WebSocket requests in seconds",
    )

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def validate_loopback_url(cls, v: str) -> str:
        """Enforce that Ollama points to a safe local loopback address."""
        normalized = v.strip().rstrip("/")
        if not (
            normalized.startswith("http://127.0.0.1")
            or normalized.startswith("http://localhost")
        ):
            raise ValueError(
                f"OLLAMA_BASE_URL must be a local loopback address (got '{v}')"
            )
        return normalized

    @field_validator("cdp_host")
    @classmethod
    def validate_cdp_host(cls, v: str) -> str:
        """Enforce that CDP host points strictly to local loopback."""
        normalized = v.strip().lower()
        if normalized not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                f"CDP_HOST must be a local loopback address (got '{v}'). Remote debugging over networks is forbidden."
            )
        return normalized

    def get_asset_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Inspect and return validation state for all declared asset files."""
        catalog: Dict[str, Dict[str, Any]] = {}
        assets_to_check = {
            "icon": self.icon_path,
            "music": self.music_path,
            "voice_clean": self.voice_clean_path,
            "voice_cond": self.voice_cond_path,
        }

        for asset_key, path in assets_to_check.items():
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            catalog[asset_key] = {
                "name": path.name,
                "path": str(path),
                "exists": exists,
                "size_bytes": size,
                "status": "ready" if (exists and size > 0) else "missing",
            }
        return catalog

    def verify_all_assets_present(self) -> bool:
        """Return True if all core assets exist and are non-empty."""
        catalog = self.get_asset_catalog()
        return all(item["status"] == "ready" for item in catalog.values())


# Global settings singleton
settings = Settings()
