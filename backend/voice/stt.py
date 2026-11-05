"""Local Speech-to-Text (STT) Module for JARVIS.

Uses local faster-whisper (base.en) for high-performance offline speech transcription.
Includes microphone recording with push-to-talk and voice-activity-based silence detection.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result of speech-to-text transcription."""
    text: str
    latency_sec: float
    audio_duration_sec: float
    is_empty: bool = False
    error: Optional[str] = None


class AudioRecorder:
    """Microphone recording utility for macOS and Apple Silicon."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        calibration_duration_sec: Optional[float] = None,
        noise_floor_multiplier: Optional[float] = None,
        min_energy_threshold: Optional[float] = None,
        max_energy_threshold: Optional[float] = None,
        silence_threshold_multiplier: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.calibration_duration_sec = (
            calibration_duration_sec
            if calibration_duration_sec is not None
            else getattr(settings, "stt_calibration_duration_sec", 0.4)
        )
        self.noise_floor_multiplier = (
            noise_floor_multiplier
            if noise_floor_multiplier is not None
            else getattr(settings, "stt_noise_floor_multiplier", 2.0)
        )
        self.min_energy_threshold = (
            min_energy_threshold
            if min_energy_threshold is not None
            else getattr(settings, "stt_min_energy_threshold", 0.008)
        )
        self.max_energy_threshold = (
            max_energy_threshold
            if max_energy_threshold is not None
            else getattr(settings, "stt_max_energy_threshold", 0.080)
        )
        self.silence_threshold_multiplier = (
            silence_threshold_multiplier
            if silence_threshold_multiplier is not None
            else getattr(settings, "stt_silence_threshold_multiplier", 1.3)
        )

    @staticmethod
    def estimate_noise_floor(energies: list[float] | np.ndarray) -> float:
        """Estimate ambient noise floor RMS from energy samples using 25th percentile.

        Using the 25th percentile effectively filters out isolated loud transients
        (clicks, pops, taps) during the calibration phase without underestimating
        steady ambient noise.
        """
        arr = np.asarray(energies, dtype=np.float32)
        if len(arr) == 0:
            return 0.005
        return float(np.percentile(arr, 25))

    def derive_speech_threshold(self, noise_floor: float) -> float:
        """Derive speech-start RMS threshold from measured noise floor.

        Applies noise_floor_multiplier and clamps the result between
        min_energy_threshold and max_energy_threshold.
        """
        raw_threshold = noise_floor * self.noise_floor_multiplier
        return float(np.clip(raw_threshold, self.min_energy_threshold, self.max_energy_threshold))

    def derive_silence_threshold(self, noise_floor: float, speech_threshold: float) -> float:
        """Derive silence endpointing RMS threshold from noise floor and speech threshold.

        Calculated as noise_floor * silence_threshold_multiplier, bounded below by
        min_energy_threshold * 0.5 (to prevent hypersensitivity in quiet rooms) and
        bounded above by speech_threshold * 0.85 (to ensure hysteresis).
        """
        raw_silence = noise_floor * self.silence_threshold_multiplier
        silence_thresh = max(raw_silence, self.min_energy_threshold * 0.5)
        max_allowed = speech_threshold * 0.85
        return float(min(silence_thresh, max_allowed))

    @staticmethod
    def get_input_devices() -> list[dict]:
        """List available audio input devices."""
        try:
            devices = sd.query_devices()
            if isinstance(devices, dict):
                devices = [devices]
            return [d for d in devices if d.get("max_input_channels", 0) > 0]
        except Exception as exc:
            logger.error("Failed to query audio input devices: %s", exc)
            return []

    def is_input_available(self) -> bool:
        """Check if at least one audio input device is available."""
        return len(self.get_input_devices()) > 0

    def record_fixed(self, duration_sec: float = 3.0) -> np.ndarray:
        """Record audio for a fixed duration.

        Args:
            duration_sec: Duration in seconds.

        Returns:
            np.ndarray: Audio samples float32 mono.
        """
        if not self.is_input_available():
            raise RuntimeError("No audio input device available.")

        frames = int(duration_sec * self.sample_rate)
        recording = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )
        sd.wait()
        return recording.flatten()

    def record_push_to_talk(
        self,
        stop_event: threading.Event,
        max_duration_sec: float = 15.0,
        chunk_duration_sec: float = 0.1,
    ) -> np.ndarray:
        """Record audio until stop_event is set or max_duration is reached.

        Args:
            stop_event: Event whose setting terminates the recording.
            max_duration_sec: Safety cap on total recording duration.
            chunk_duration_sec: Duration of each chunk read.

        Returns:
            np.ndarray: Audio samples float32 mono.
        """
        if not self.is_input_available():
            raise RuntimeError("No audio input device available.")

        chunk_size = int(self.sample_rate * chunk_duration_sec)
        max_chunks = int(max_duration_sec / chunk_duration_sec)
        chunks: list[np.ndarray] = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=chunk_size,
            ) as stream:
                count = 0
                while not stop_event.is_set() and count < max_chunks:
                    data, overflowed = stream.read(chunk_size)
                    if overflowed:
                        logger.debug("Microphone input buffer overflowed.")
                    chunks.append(data.copy())
                    count += 1
        except Exception as exc:
            logger.error("Audio recording error in push-to-talk: %s", exc)
            raise RuntimeError(f"Microphone recording failed: {exc}") from exc

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        combined = np.concatenate(chunks, axis=0)
        return combined.flatten()

    def record_until_silence(
        self,
        silence_duration_sec: float = 1.2,
        max_duration_sec: float = 15.0,
        energy_threshold: float = 0.015,
        speech_lead_chunks: int = 3,
        calibration_duration_sec: Optional[float] = None,
        dynamic_calibration: bool = True,
    ) -> np.ndarray:
        """Record audio with automatic speech detection and silence endpointing.

        Uses dynamic noise-floor calibration to estimate ambient noise and set
        proportional speech and silence thresholds. Outlier-robust estimation
        (25th percentile) ensures transient noise does not inflate the floor.

        Args:
            silence_duration_sec: Seconds of silence after speech to conclude recording.
            max_duration_sec: Maximum recording duration.
            energy_threshold: Fallback/fixed RMS amplitude threshold (used if dynamic_calibration is False).
            speech_lead_chunks: Pre-speech chunks to preserve context before trigger.
            calibration_duration_sec: Optional duration for initial noise calibration.
            dynamic_calibration: If True, derives thresholds from ambient noise calibration.

        Returns:
            np.ndarray: Audio samples float32 mono.
        """
        if not self.is_input_available():
            raise RuntimeError("No audio input device available.")

        chunk_duration_sec = 0.1
        chunk_size = int(self.sample_rate * chunk_duration_sec)
        silence_chunks_needed = int(round(silence_duration_sec / chunk_duration_sec))
        max_chunks = max(1, int(round(max_duration_sec / chunk_duration_sec)))

        calib_duration = (
            calibration_duration_sec
            if calibration_duration_sec is not None
            else self.calibration_duration_sec
        )
        calib_chunks = max(2, int(round(calib_duration / chunk_duration_sec)))
        if calib_chunks > max_chunks:
            calib_chunks = max_chunks

        buffer: list[np.ndarray] = []
        calib_energies: list[float] = []
        speech_started = False
        silence_count = 0

        effective_speech_thresh = energy_threshold
        effective_silence_thresh = energy_threshold
        calibrated = not dynamic_calibration

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=chunk_size,
            ) as stream:
                for chunk_idx in range(max_chunks):
                    data, _ = stream.read(chunk_size)
                    if data is None or len(data) == 0:
                        continue
                    rms = float(np.sqrt(np.mean(data**2)))
                    buffer.append(data.copy())

                    # Calibration phase
                    if not calibrated:
                        calib_energies.append(rms)
                        if len(calib_energies) >= calib_chunks:
                            noise_floor = self.estimate_noise_floor(calib_energies)
                            effective_speech_thresh = self.derive_speech_threshold(noise_floor)
                            effective_silence_thresh = self.derive_silence_threshold(
                                noise_floor, effective_speech_thresh
                            )
                            calibrated = True
                            logger.debug(
                                "STT noise calibration complete: floor=%.4f, speech_thresh=%.4f, silence_thresh=%.4f",
                                noise_floor,
                                effective_speech_thresh,
                                effective_silence_thresh,
                            )

                            # Check if speech occurred during calibration
                            high_indices = [
                                i for i, e in enumerate(calib_energies)
                                if e >= effective_speech_thresh
                            ]
                            # Only treat as speech onset if there are multiple high chunks
                            # or the last chunk of calibration is high (potential speech continuation)
                            if len(high_indices) >= 2:
                                speech_started = True
                                first_high = high_indices[0]
                                lead_start = max(0, first_high - speech_lead_chunks)
                                buffer = buffer[lead_start:]
                                silence_count = 0
                            elif len(high_indices) == 1 and high_indices[0] == len(calib_energies) - 1:
                                # Last chunk was high; keep lead chunks and evaluate next chunk
                                buffer = buffer[-speech_lead_chunks:]
                            else:
                                # Isolated transient or all quiet: keep up to speech_lead_chunks
                                buffer = buffer[-speech_lead_chunks:]
                        continue

                    # Active detection phase
                    if not speech_started:
                        if rms >= effective_speech_thresh:
                            speech_started = True
                            # Keep lead chunks before speech onset plus this triggering chunk
                            buffer = buffer[-(speech_lead_chunks + 1):]
                            silence_count = 0
                        else:
                            # Prune pre-speech silence buffer to preserve lead window
                            if len(buffer) > speech_lead_chunks:
                                buffer.pop(0)
                    else:
                        if rms < effective_silence_thresh:
                            silence_count += 1
                            if silence_count >= silence_chunks_needed:
                                logger.debug(
                                    "Silence endpoint reached (%d chunks), ending recording",
                                    silence_count,
                                )
                                break
                        else:
                            silence_count = 0

        except Exception as exc:
            logger.error("Audio recording error in silence detection: %s", exc)
            raise RuntimeError(f"Microphone recording failed: {exc}") from exc

        if not speech_started or not buffer:
            return np.zeros(0, dtype=np.float32)

        combined = np.concatenate(buffer, axis=0)
        return combined.flatten()


class SpeechToTextEngine:
    """Local Speech-to-Text Engine backed by faster-whisper."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.stt_model_name
        self.device = device or settings.stt_device
        self.compute_type = compute_type or settings.stt_compute_type
        self._model = None
        self._lock = threading.RLock()


    @property
    def model(self):
        """Lazy load and return the WhisperModel instance."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self.load()
        return self._model

    def load(self) -> None:
        """Explicitly load the local WhisperModel into memory."""
        t0 = time.perf_counter()
        logger.info(
            "Loading local STT model '%s' on %s (%s)...",
            self.model_name,
            self.device,
            self.compute_type,
        )
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            load_sec = time.perf_counter() - t0
            logger.info("STT model loaded successfully in %.2fs.", load_sec)
        except Exception as exc:
            logger.error("Failed to load STT model '%s': %s", self.model_name, exc)
            raise RuntimeError(f"Failed to load Whisper STT model: {exc}") from exc

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe in-memory audio samples to English text.

        Args:
            audio: 1D numpy array of audio samples (float32).
            sample_rate: Sample rate of the audio (default 16000).

        Returns:
            STTResult: Result containing text, latency, and duration.
        """
        t0 = time.perf_counter()

        # Check for empty or near-silent audio
        if audio is None or len(audio) == 0:
            return STTResult(
                text="",
                latency_sec=time.perf_counter() - t0,
                audio_duration_sec=0.0,
                is_empty=True,
            )

        # Ensure float32 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Audio duration calculation
        audio_duration_sec = len(audio) / sample_rate

        # Minimum duration check (e.g. at least 0.2s of audio)
        if audio_duration_sec < 0.2:
            return STTResult(
                text="",
                latency_sec=time.perf_counter() - t0,
                audio_duration_sec=audio_duration_sec,
                is_empty=True,
            )

        # Silence check by RMS amplitude
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.003:
            return STTResult(
                text="",
                latency_sec=time.perf_counter() - t0,
                audio_duration_sec=audio_duration_sec,
                is_empty=True,
            )

        # Resample to 16000 Hz if necessary
        if sample_rate != 16000:
            target_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, target_samples).astype(np.float32)

        try:
            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            transcribed_text = " ".join(seg.text for seg in segments).strip()
            latency_sec = time.perf_counter() - t0

            return STTResult(
                text=transcribed_text,
                latency_sec=latency_sec,
                audio_duration_sec=audio_duration_sec,
                is_empty=(len(transcribed_text) == 0),
            )
        except Exception as exc:
            latency_sec = time.perf_counter() - t0
            logger.error("STT transcription failed: %s", exc)
            return STTResult(
                text="",
                latency_sec=latency_sec,
                audio_duration_sec=audio_duration_sec,
                is_empty=True,
                error=str(exc),
            )

    def transcribe_file(self, file_path: Path | str) -> STTResult:
        """Transcribe an audio file using soundfile for safe container parsing.

        Args:
            file_path: Path to the audio file (.wav, etc.).

        Returns:
            STTResult: Result containing text, latency, and duration.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            return STTResult(
                text="",
                latency_sec=0.0,
                audio_duration_sec=0.0,
                is_empty=True,
                error=f"File not found: {path}",
            )

        try:
            audio_data, sr = sf.read(str(path), dtype="float32")
            return self.transcribe_audio(audio_data, sample_rate=sr)
        except Exception as exc:
            logger.error("Failed to read audio file %s: %s", path, exc)
            return STTResult(
                text="",
                latency_sec=0.0,
                audio_duration_sec=0.0,
                is_empty=True,
                error=f"Audio read failure: {exc}",
            )
