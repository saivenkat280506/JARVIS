"""Local Text-to-Speech (TTS) Module for JARVIS.

Uses local pocket-tts with conditioned voice cloning on assets/voices/jarvis_voice_clean.wav.
Provides clean text synthesis, temporary file generation, safe macOS afplay playback,
and guaranteed cleanup of temporary audio files.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of text-to-speech synthesis and playback."""
    text: str
    latency_sec: float
    audio_duration_sec: float
    played: bool = False
    error: Optional[str] = None
    playback_duration_sec: Optional[float] = None
    playback_start_timestamp: Optional[float] = None


def sanitize_text_for_speech(raw_text: str) -> str:
    """Normalize text into clean conversational English for natural speech synthesis.

    Strips markdown artifacts, code blocks, excessive symbols, and chain-of-thought tags.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Strip reasoning tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)

    # Strip markdown code blocks ```...```
    text = re.sub(r"```[\w]*\n?.*?```", "code block omitted.", text, flags=re.DOTALL)

    # Strip inline code `code`
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Strip markdown headers (e.g. ### Header)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Strip markdown bold/italics (***text***, **text**, *text*, _text_)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Strip markdown links [link text](url) -> link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Strip bullet points and list markers at line start
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Strip technical IDs in parentheses, e.g. (ID: rem_68558e82) or (id: 12345)
    text = re.sub(r"\s*\((?:ID:?\s*|id:?\s*)?[a-zA-Z0-9_\-]{6,}\)", "", text)

    # Strip UTC timezone offsets, e.g. (UTC+05:30)
    text = re.sub(r"\s*\(UTC[+-][\d:]+\)", "", text)

    # Strip empty parentheses ()
    text = re.sub(r"\(\s*\)", "", text)

    # Clean up JSON or curly brace blocks if any
    text = re.sub(r"\{[^{}]*\}", "", text)

    # Replace underscores in file names/identifiers with spaces for natural speech
    text = re.sub(r"(\w)_(\w)", r"\1 \2", text)

    # Strip stray asterisks
    text = text.replace("*", "")

    # Normalize multiple whitespaces and newlines into clean spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



class TextToSpeechEngine:
    """Local Text-to-Speech Engine backed by Pocket TTS."""

    def __init__(
        self,
        voice_prompt_path: Path | str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.voice_prompt_path = Path(voice_prompt_path or settings.voice_clean_path).resolve()
        self.sample_rate = sample_rate or settings.tts_sample_rate
        self._model = None
        self._voice_state = None
        self._lock = threading.RLock()
        self._current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()


    @property
    def model(self):
        """Lazy load and return the TTSModel instance."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self.load()
        return self._model

    def load(self) -> None:
        """Load the TTS model and initialize voice conditioning state."""
        t0 = time.perf_counter()
        logger.info("Loading Pocket TTS model...")
        try:
            from pocket_tts import TTSModel

            self._model = TTSModel.load_model()
            load_sec = time.perf_counter() - t0
            logger.info("Pocket TTS loaded in %.2fs (sample_rate=%d)", load_sec, self._model.sample_rate)

            # Compute voice conditioning state from clean voice sample
            if not self.voice_prompt_path.is_file():
                raise FileNotFoundError(f"Voice sample not found at: {self.voice_prompt_path}")

            t_cond = time.perf_counter()
            self._voice_state = self._model.get_state_for_audio_prompt(
                str(self.voice_prompt_path),
                truncate=True,
            )
            logger.info("Voice conditioning extracted in %.2fs", time.perf_counter() - t_cond)

        except Exception as exc:
            logger.error("Failed to initialize Pocket TTS engine: %s", exc)
            raise RuntimeError(f"TTS Engine initialization failed: {exc}") from exc

    def synthesize(self, text: str) -> tuple[np.ndarray, int, float]:
        """Synthesize text to audio samples.

        Args:
            text: Text to synthesize.

        Returns:
            tuple[np.ndarray, int, float]: (audio_samples, sample_rate, latency_sec)
        """
        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return np.zeros(0, dtype=np.float32), self.sample_rate, 0.0

        t0 = time.perf_counter()
        with self._lock:
            # Ensure model and voice state are loaded
            _ = self.model
            if self._voice_state is None:
                self._voice_state = self._model.get_state_for_audio_prompt(
                    str(self.voice_prompt_path),
                    truncate=True,
                )

            # Generate audio tensor
            audio_tensor: torch.Tensor = self._model.generate_audio(
                self._voice_state,
                clean_text,
                copy_state=True,
            )

        latency_sec = time.perf_counter() - t0
        audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32)
        sr = getattr(self._model, "sample_rate", self.sample_rate)
        return audio_np, sr, latency_sec

    def speak(self, text: str) -> TTSResult:
        """Synthesize text, write to a safe temp file, play via afplay, and clean up.

        Args:
            text: Text to speak.

        Returns:
            TTSResult: Details on synthesis latency, audio duration, and playback status.
        """
        clean_text = sanitize_text_for_speech(text)
        if not clean_text:
            return TTSResult(
                text=text,
                latency_sec=0.0,
                audio_duration_sec=0.0,
                played=False,
            )

        try:
            audio_np, sr, latency_sec = self.synthesize(clean_text)
            if len(audio_np) == 0:
                return TTSResult(
                    text=clean_text,
                    latency_sec=latency_sec,
                    audio_duration_sec=0.0,
                    played=False,
                )

            audio_duration_sec = len(audio_np) / sr

            # Write to a secure temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(prefix="jarvis_speech_", suffix=".wav")
            os.close(temp_fd)

            try:
                sf.write(temp_path, audio_np, sr)

                # Play via macOS built-in afplay
                t_play_start = time.perf_counter()
                played = self._play_audio_file(temp_path)
                playback_duration_sec = time.perf_counter() - t_play_start if played else None

                return TTSResult(
                    text=clean_text,
                    latency_sec=latency_sec,
                    audio_duration_sec=audio_duration_sec,
                    played=played,
                    playback_duration_sec=playback_duration_sec,
                    playback_start_timestamp=t_play_start if played else None,
                )
            finally:
                # Guaranteed cleanup of temporary audio file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError as err:
                        logger.warning("Failed to remove temporary audio file %s: %s", temp_path, err)

        except Exception as exc:
            logger.error("TTS synthesis or playback failed: %s", exc)
            return TTSResult(
                text=clean_text,
                latency_sec=0.0,
                audio_duration_sec=0.0,
                played=False,
                error=str(exc),
            )

    def _play_audio_file(self, wav_path: str) -> bool:
        """Play a WAV file via /usr/bin/afplay and wait for completion."""
        afplay_bin = "/usr/bin/afplay"
        if not os.path.isfile(afplay_bin):
            logger.error("macOS afplay utility not found at %s", afplay_bin)
            return False

        try:
            with self._process_lock:
                self._current_process = subprocess.Popen(
                    [afplay_bin, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Wait for playback to finish
            self._current_process.wait()
            return self._current_process.returncode == 0
        except Exception as exc:
            logger.error("afplay execution failed: %s", exc)
            return False
        finally:
            with self._process_lock:
                self._current_process = None

    def stop_playback(self) -> None:
        """Terminate active speech playback process if currently running."""
        with self._process_lock:
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                    self._current_process.wait(timeout=0.5)
                except Exception as exc:
                    logger.debug("Failed to terminate active afplay process: %s", exc)
                finally:
                    self._current_process = None
