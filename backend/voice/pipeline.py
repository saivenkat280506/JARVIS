"""Local Voice Interaction Pipeline for JARVIS.

Coordinates:
  Microphone Input → STT (faster-whisper) → Agent Runtime + Tools → TTS (Pocket TTS) → Audio Playback

Enforces:
  - Complete local offline execution.
  - Strict audio coordination: recording is disabled during TTS playback.
  - Music playback awareness: pauses music during voice turn if active.
  - No reasoning/thinking chains or raw tool JSON exposed to TTS.
  - Detailed component and end-to-end latency measurements.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import difflib
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from agent.runtime import AgentRunResult, AgentRuntime
from agent.graph_runtime import LangGraphAgentRuntime
from config.settings import settings
from tools.music import MusicController, MusicState, default_music_controller
from voice.stt import AudioRecorder, SpeechToTextEngine, STTResult
from voice.tts import TextToSpeechEngine, TTSResult

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_SIMILARITY_THRESHOLD = 0.85
MIN_REFLECTION_CHARS = 15
MIN_REFLECTION_WORDS = 3

SHORT_ACKNOWLEDGMENT_WORDS = {
    "yes", "no", "ok", "okay", "yep", "nope", "yeah", "nah",
    "thanks", "thank you", "got it", "understood", "confirmed",
    "stop", "cancel", "help", "repeat", "next", "continue",
    "proceed", "sure", "alright", "right", "correct",
}


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for reflection comparison by lowercasing,
    stripping punctuation, and collapsing whitespace.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_transcript_reflection(
    candidate: str,
    previous_response: Optional[str],
    similarity_threshold: float = DEFAULT_REFLECTION_SIMILARITY_THRESHOLD,
) -> bool:
    """Determine whether candidate transcript is an echo/reflection of previous assistant response.

    Conservative guard requirements:
    1. Previous assistant response must exist and have sufficient length.
    2. Candidate must contain enough meaningful text (>= MIN_REFLECTION_WORDS and >= MIN_REFLECTION_CHARS).
    3. Candidate must not be a short acknowledgment or ordinary control word.
    4. Sequence similarity ratio (difflib.SequenceMatcher) between normalized texts must meet or exceed threshold.
    """
    if not candidate or not previous_response:
        return False

    norm_candidate = normalize_text_for_comparison(candidate)
    norm_prev = normalize_text_for_comparison(previous_response)

    # Candidate must have meaningful length to avoid false positives on short commands
    words = norm_candidate.split()
    if len(norm_candidate) < MIN_REFLECTION_CHARS or len(words) < MIN_REFLECTION_WORDS:
        return False

    # Short acknowledgments / common control commands are never suppressed
    if norm_candidate in SHORT_ACKNOWLEDGMENT_WORDS:
        return False

    # Previous response must also have sufficient meaningful text
    if len(norm_prev) < MIN_REFLECTION_CHARS:
        return False

    # Compute sequence similarity ratio
    matcher = difflib.SequenceMatcher(None, norm_candidate, norm_prev)
    ratio = matcher.ratio()

    # Also check if candidate is a substantial substring of previous response
    # E.g., if Whisper clipped the first word or two of a long assistant response:
    if norm_candidate in norm_prev:
        sub_ratio = len(norm_candidate) / len(norm_prev)
        if sub_ratio >= similarity_threshold:
            return True

    return ratio >= similarity_threshold


@dataclass
class VoiceTurnMetrics:
    """Structured per-stage latency metrics for a single voice interaction turn.

    All elapsed time measurements use a monotonic clock (time.perf_counter()).
    Unavailable stages are explicitly represented as None.
    """
    turn_id: str
    turn_type: str  # 'voice', 'push_to_talk', 'audio'
    success: bool
    error: Optional[str] = None

    # Stage 1: Audio capture
    audio_capture_ms: Optional[float] = None
    user_audio_duration_sec: float = 0.0

    # Stage 2: STT transcription
    stt_transcription_ms: Optional[float] = None

    # Stage 3: Agent runtime & tool execution
    agent_runtime_ms: Optional[float] = None
    pure_llm_ms: Optional[float] = None
    tool_execution_ms: Optional[float] = None

    # Stage 4: TTS synthesis
    tts_synthesis_ms: Optional[float] = None
    tts_audio_duration_sec: Optional[float] = None

    # Stage 5: Time to first audio playback (TTFA)
    # Measured from end of capture (or turn start for pre-recorded audio)
    # to the exact start of audio playback (afplay launch).
    ttfa_ms: Optional[float] = None

    # Stage 6: Audio playback
    audio_playback_ms: Optional[float] = None

    # Stage 7: Post-playback acoustic cooldown
    post_playback_cooldown_ms: Optional[float] = None

    # Stage 8: Total turn wall-clock duration
    total_turn_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a sanitized dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "turn_type": self.turn_type,
            "success": self.success,
            "error": self.error,
            "audio_capture_ms": round(self.audio_capture_ms, 2) if self.audio_capture_ms is not None else None,
            "user_audio_duration_sec": round(self.user_audio_duration_sec, 3),
            "stt_transcription_ms": round(self.stt_transcription_ms, 2) if self.stt_transcription_ms is not None else None,
            "agent_runtime_ms": round(self.agent_runtime_ms, 2) if self.agent_runtime_ms is not None else None,
            "pure_llm_ms": round(self.pure_llm_ms, 2) if self.pure_llm_ms is not None else None,
            "tool_execution_ms": round(self.tool_execution_ms, 2) if self.tool_execution_ms is not None else None,
            "tts_synthesis_ms": round(self.tts_synthesis_ms, 2) if self.tts_synthesis_ms is not None else None,
            "tts_audio_duration_sec": round(self.tts_audio_duration_sec, 3) if self.tts_audio_duration_sec is not None else None,
            "ttfa_ms": round(self.ttfa_ms, 2) if self.ttfa_ms is not None else None,
            "audio_playback_ms": round(self.audio_playback_ms, 2) if self.audio_playback_ms is not None else None,
            "post_playback_cooldown_ms": round(self.post_playback_cooldown_ms, 2) if self.post_playback_cooldown_ms is not None else None,
            "total_turn_ms": round(self.total_turn_ms, 2),
        }

    def to_compact_log(self) -> str:
        """Format metrics as a single-line sanitized structured string."""
        def fmt(val: Optional[float], unit: str = "ms") -> str:
            if val is None:
                return "N/A"
            return f"{val:.1f}{unit}"

        def fmt_s(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"{val:.2f}s"

        return (
            f"turn_id={self.turn_id} type={self.turn_type} "
            f"capture={fmt(self.audio_capture_ms)} audio_dur={fmt_s(self.user_audio_duration_sec)} "
            f"stt={fmt(self.stt_transcription_ms)} runtime={fmt(self.agent_runtime_ms)} "
            f"(llm={fmt(self.pure_llm_ms)}, tools={fmt(self.tool_execution_ms)}) "
            f"tts_synth={fmt(self.tts_synthesis_ms)} tts_dur={fmt_s(self.tts_audio_duration_sec)} "
            f"ttfa={fmt(self.ttfa_ms)} playback={fmt(self.audio_playback_ms)} "
            f"cooldown={fmt(self.post_playback_cooldown_ms)} "
            f"total={fmt(self.total_turn_ms)} success={self.success}"
            + (f" error={self.error}" if self.error else "")
        )


@dataclass
class VoiceTurnResult:
    """Detailed outcome of a single voice interaction turn."""
    user_query: str
    final_response: str
    conversation_id: Optional[str] = None
    tool_records: list = field(default_factory=list)
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    pure_llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    audio_duration_sec: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metrics: Optional[VoiceTurnMetrics] = None


@dataclass
class _TurnLifecycleContext:
    """Internal context tracking state across a single managed voice turn."""
    music_was_playing: bool = False
    music_altered_by_action: bool = False


class VoicePipeline:
    """Orchestrates end-to-end local voice interaction for JARVIS."""

    def __init__(
        self,
        runtime: AgentRuntime | LangGraphAgentRuntime,
        stt_engine: Optional[SpeechToTextEngine] = None,
        tts_engine: Optional[TextToSpeechEngine] = None,
        recorder: Optional[AudioRecorder] = None,
        conversation_id: Optional[str] = None,
        music_controller: Optional[MusicController] = None,
        post_playback_cooldown_sec: Optional[float] = None,
        reflection_similarity_threshold: Optional[float] = None,
    ) -> None:
        self.runtime = runtime
        self.stt_engine = stt_engine or SpeechToTextEngine()
        self.tts_engine = tts_engine or TextToSpeechEngine()
        self.recorder = recorder or AudioRecorder()
        self.conversation_id = conversation_id
        self._music_controller = music_controller
        self.post_playback_cooldown_sec = (
            post_playback_cooldown_sec
            if post_playback_cooldown_sec is not None
            else getattr(settings, "voice_post_playback_cooldown_sec", 0.35)
        )
        self.reflection_similarity_threshold = (
            reflection_similarity_threshold
            if reflection_similarity_threshold is not None
            else getattr(settings, "voice_reflection_similarity_threshold", DEFAULT_REFLECTION_SIMILARITY_THRESHOLD)
        )
        self._lock = asyncio.Lock()
        self._last_spoken_response: Optional[str] = None
        self._last_playback_time: float = 0.0

    @property
    def music_controller(self) -> MusicController:
        return self._music_controller if self._music_controller is not None else default_music_controller

    @music_controller.setter
    def music_controller(self, controller: Optional[MusicController]) -> None:
        self._music_controller = controller

    def reset_session(self) -> None:
        """Reset in-memory voice session state including the reflection guard."""
        self._last_spoken_response = None
        self._last_playback_time = 0.0

    def _is_reflection(self, query: str) -> bool:
        """Determine whether candidate text is an echo of the previous spoken response."""
        return is_transcript_reflection(
            candidate=query,
            previous_response=self._last_spoken_response,
            similarity_threshold=self.reflection_similarity_threshold,
        )

    async def _apply_post_playback_cooldown(self) -> None:
        """Enforce post-playback acoustic cooldown after assistant TTS playback."""
        if self.post_playback_cooldown_sec > 0:
            logger.debug(
                "Acoustic post-playback cooldown: waiting %.3fs before allowing next recording",
                self.post_playback_cooldown_sec,
            )
            await asyncio.sleep(self.post_playback_cooldown_sec)

    async def _wait_pre_recording_cooldown(self) -> float:
        """Ensure post_playback_cooldown_sec has elapsed since last playback before recording starts."""
        if self._last_playback_time > 0 and self.post_playback_cooldown_sec > 0:
            elapsed = time.monotonic() - self._last_playback_time
            remaining = self.post_playback_cooldown_sec - elapsed
            if remaining > 0:
                logger.debug(
                    "Residual post-playback cooldown: waiting %.3fs before recording",
                    remaining,
                )
                await asyncio.sleep(remaining)
                return remaining
        return 0.0

    def warmup(self) -> dict[str, float]:
        """Preload STT and TTS models into memory and return warmup latencies."""
        logger.info("Warming up local STT and TTS engines...")
        t0 = time.perf_counter()
        self.stt_engine.load()
        stt_load_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.tts_engine.load()
        tts_load_sec = time.perf_counter() - t1

        return {
            "stt_warmup_sec": stt_load_sec,
            "tts_warmup_sec": tts_load_sec,
            "total_warmup_sec": time.perf_counter() - t0,
        }

    @asynccontextmanager
    async def _turn_lifecycle(self):
        """Unified lifecycle context for voice turns.

        Acquires `self._lock` to serialize microphone capture, transcription,
        agent inference, and audio playback.

        Inspects music state: if playing before the turn, pauses playback.
        Ensures music remains paused across recording, STT, LLM inference, tool
        execution, and TTS playback.

        On turn completion, exception, or cancellation:
        Resumes playback only if music was playing prior to the turn and remains
        paused (i.e. not stopped or altered by tools), and releases `self._lock`.
        """
        async with self._lock:
            controller = self.music_controller
            music_was_playing = False
            try:
                music_was_playing = (controller.get_state() == MusicState.PLAYING)
            except Exception as exc:
                logger.warning("Failed to inspect music controller state: %s", exc)

            if music_was_playing:
                try:
                    res = controller.pause()
                    if isinstance(res, dict) and not res.get("success", True):
                        logger.warning("Music pause returned failure status: %s", res.get("error"))
                except Exception as exc:
                    logger.warning("Failed to pause music playback before voice turn: %s", exc)

            ctx = _TurnLifecycleContext(music_was_playing=music_was_playing)
            try:
                yield ctx
            finally:
                if music_was_playing and not ctx.music_altered_by_action:
                    try:
                        current_state = controller.get_state()
                        if current_state == MusicState.PAUSED:
                            res = controller.resume()
                            if isinstance(res, dict) and not res.get("success", True):
                                logger.warning("Music resume returned failure status: %s", res.get("error"))
                    except Exception as exc:
                        logger.warning("Failed to resume music playback after voice turn: %s", exc)

    async def _execute_turn_from_audio_core(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        speak_response: bool = True,
        conversation_id: Optional[str] = None,
        lifecycle_context: Optional[_TurnLifecycleContext] = None,
        turn_id: Optional[str] = None,
        turn_type: str = "audio",
        turn_start: Optional[float] = None,
        audio_capture_ms: Optional[float] = None,
        ttfa_reference_start: Optional[float] = None,
    ) -> VoiceTurnResult:
        """Core voice turn processing from audio samples (unlocked)."""
        effective_turn_id = turn_id or f"turn_{uuid.uuid4().hex[:8]}"
        effective_turn_start = turn_start if turn_start is not None else time.perf_counter()
        effective_ttfa_ref = ttfa_reference_start if ttfa_reference_start is not None else effective_turn_start
        effective_cid = conversation_id if conversation_id is not None else self.conversation_id
        user_audio_duration_sec = 0.0
        try:
            sr_num = float(sample_rate)
            if audio_data is not None and len(audio_data) > 0 and sr_num > 0:
                user_audio_duration_sec = float(len(audio_data)) / sr_num
        except (TypeError, ValueError):
            user_audio_duration_sec = 0.0

        stt_transcription_ms: Optional[float] = None
        stt_latency_ms: float = 0.0
        llm_latency_ms: Optional[float] = None
        tool_latency_ms: Optional[float] = None
        pure_llm_latency_ms: Optional[float] = None
        tts_latency_ms: float = 0.0
        tts_synthesis_ms: Optional[float] = None
        tts_audio_duration_sec: Optional[float] = None
        ttfa_ms: Optional[float] = None
        audio_playback_ms: Optional[float] = None
        post_playback_cooldown_ms: Optional[float] = None
        audio_dur_sec: float = 0.0

        try:
            # 1. Speech-to-Text
            t_stt_start = time.perf_counter()
            try:
                stt_res: STTResult = await asyncio.to_thread(
                    self.stt_engine.transcribe_audio,
                    audio_data,
                    sample_rate,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                t_stt_end = time.perf_counter()
                stt_transcription_ms = (t_stt_end - t_stt_start) * 1000.0
                total_duration_ms = (time.perf_counter() - effective_turn_start) * 1000.0
                logger.error("STT transcription failure: %s", exc)
                metrics = VoiceTurnMetrics(
                    turn_id=effective_turn_id,
                    turn_type=turn_type,
                    success=False,
                    error=f"Speech recognition error: {exc}",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=user_audio_duration_sec,
                    stt_transcription_ms=stt_transcription_ms,
                    total_turn_ms=total_duration_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query="",
                    final_response="",
                    conversation_id=effective_cid,
                    total_latency_ms=total_duration_ms,
                    success=False,
                    error=f"Speech recognition error: {exc}",
                    metrics=metrics,
                )

            t_stt_end = time.perf_counter()
            stt_latency_ms = stt_res.latency_sec * 1000.0
            stt_transcription_ms = (t_stt_end - t_stt_start) * 1000.0

            if stt_res.is_empty or not stt_res.text.strip():
                total_duration_ms = (time.perf_counter() - effective_turn_start) * 1000.0
                metrics = VoiceTurnMetrics(
                    turn_id=effective_turn_id,
                    turn_type=turn_type,
                    success=True,
                    error="No speech detected",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=user_audio_duration_sec,
                    stt_transcription_ms=stt_transcription_ms,
                    total_turn_ms=total_duration_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query="",
                    final_response="",
                    conversation_id=effective_cid,
                    stt_latency_ms=stt_latency_ms,
                    total_latency_ms=total_duration_ms,
                    success=True,
                    error="No speech detected",
                    metrics=metrics,
                )

            query = stt_res.text.strip()

            # Check transcript reflection guard before invoking AgentRuntime
            if self._is_reflection(query):
                logger.info("Transcript suppressed by reflection guard (matches previous assistant output)")
                total_duration_ms = (time.perf_counter() - effective_turn_start) * 1000.0
                metrics = VoiceTurnMetrics(
                    turn_id=effective_turn_id,
                    turn_type=turn_type,
                    success=True,
                    error="Transcript suppressed by reflection guard",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=user_audio_duration_sec,
                    stt_transcription_ms=stt_transcription_ms,
                    total_turn_ms=total_duration_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query=query,
                    final_response="",
                    conversation_id=effective_cid,
                    stt_latency_ms=stt_latency_ms,
                    total_latency_ms=total_duration_ms,
                    success=True,
                    error="Transcript suppressed by reflection guard",
                    metrics=metrics,
                )

            # 2. Agent Runtime & Authorized Tool Execution
            llm_start = time.perf_counter()
            try:
                if effective_cid is not None:
                    agent_result: AgentRunResult = await self.runtime.run(
                        query,
                        conversation_id=effective_cid,
                    )
                else:
                    agent_result: AgentRunResult = await self.runtime.run(query)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                llm_end = time.perf_counter()
                llm_latency_ms = (llm_end - llm_start) * 1000.0
                total_duration_ms = (time.perf_counter() - effective_turn_start) * 1000.0
                logger.error("Agent runtime execution failure: %s", exc)
                metrics = VoiceTurnMetrics(
                    turn_id=effective_turn_id,
                    turn_type=turn_type,
                    success=False,
                    error=f"Agent inference failure: {exc}",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=user_audio_duration_sec,
                    stt_transcription_ms=stt_transcription_ms,
                    agent_runtime_ms=llm_latency_ms,
                    total_turn_ms=total_duration_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query=query,
                    final_response="",
                    conversation_id=effective_cid,
                    stt_latency_ms=stt_latency_ms,
                    llm_latency_ms=llm_latency_ms,
                    total_latency_ms=total_duration_ms,
                    success=False,
                    error=f"Agent inference failure: {exc}",
                    metrics=metrics,
                )

            llm_end = time.perf_counter()
            llm_latency_ms = (llm_end - llm_start) * 1000.0
            tool_latency_ms = sum(r.duration_sec * 1000.0 for r in agent_result.tool_records)
            pure_llm_latency_ms = max(0.0, llm_latency_ms - tool_latency_ms)

            # Check if tools altered music state (e.g. pause_music or stop_music)
            if lifecycle_context is not None and agent_result.tool_records:
                for record in agent_result.tool_records:
                    if record.tool_name in ("pause_music", "stop_music"):
                        lifecycle_context.music_altered_by_action = True
                        break

            final_response = agent_result.final_response

            # 3. Text-to-Speech & Safe Audio Playback
            if speak_response and final_response:
                try:
                    t_tts_call_start = time.perf_counter()
                    tts_res: TTSResult = await asyncio.to_thread(
                        self.tts_engine.speak,
                        final_response,
                    )
                    t_tts_call_end = time.perf_counter()
                    tts_latency_ms = tts_res.latency_sec * 1000.0
                    tts_synthesis_ms = tts_latency_ms
                    audio_dur_sec = tts_res.audio_duration_sec
                    tts_audio_duration_sec = audio_dur_sec

                    if tts_res.played:
                        # Update in-memory session reflection reference upon successful playback
                        self._last_spoken_response = final_response
                        self._last_playback_time = time.monotonic()

                        # Audio playback duration
                        if tts_res.playback_duration_sec is not None:
                            audio_playback_ms = tts_res.playback_duration_sec * 1000.0
                        else:
                            audio_playback_ms = max(
                                0.0,
                                (t_tts_call_end - t_tts_call_start) * 1000.0 - tts_latency_ms,
                            )

                        # TTFA calculation from documented reference start
                        if tts_res.playback_start_timestamp is not None:
                            ttfa_ms = (tts_res.playback_start_timestamp - effective_ttfa_ref) * 1000.0
                        else:
                            ttfa_ms = None

                        # Acoustic cooldown
                        t_cd_start = time.perf_counter()
                        await self._apply_post_playback_cooldown()
                        post_playback_cooldown_ms = (time.perf_counter() - t_cd_start) * 1000.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("TTS playback failure: %s", exc)

            total_latency_ms = (time.perf_counter() - effective_turn_start) * 1000.0

            metrics = VoiceTurnMetrics(
                turn_id=effective_turn_id,
                turn_type=turn_type,
                success=agent_result.completed,
                error=agent_result.error_message,
                audio_capture_ms=audio_capture_ms,
                user_audio_duration_sec=user_audio_duration_sec,
                stt_transcription_ms=stt_transcription_ms,
                agent_runtime_ms=llm_latency_ms,
                pure_llm_ms=pure_llm_latency_ms,
                tool_execution_ms=tool_latency_ms,
                tts_synthesis_ms=tts_synthesis_ms,
                tts_audio_duration_sec=tts_audio_duration_sec,
                ttfa_ms=ttfa_ms,
                audio_playback_ms=audio_playback_ms,
                post_playback_cooldown_ms=post_playback_cooldown_ms,
                total_turn_ms=total_latency_ms,
            )
            logger.info("Voice turn metrics: %s", metrics.to_compact_log())

            return VoiceTurnResult(
                user_query=query,
                final_response=final_response,
                conversation_id=agent_result.conversation_id or effective_cid,
                tool_records=agent_result.tool_records,
                stt_latency_ms=stt_latency_ms,
                llm_latency_ms=llm_latency_ms or 0.0,
                pure_llm_latency_ms=pure_llm_latency_ms or 0.0,
                tool_latency_ms=tool_latency_ms or 0.0,
                tts_latency_ms=tts_latency_ms,
                total_latency_ms=total_latency_ms,
                audio_duration_sec=audio_dur_sec,
                success=agent_result.completed,
                error=agent_result.error_message,
                metrics=metrics,
            )
        except asyncio.CancelledError:
            total_duration_ms = (time.perf_counter() - effective_turn_start) * 1000.0
            metrics = VoiceTurnMetrics(
                turn_id=effective_turn_id,
                turn_type=turn_type,
                success=False,
                error="Voice turn cancelled",
                audio_capture_ms=audio_capture_ms,
                user_audio_duration_sec=user_audio_duration_sec,
                stt_transcription_ms=stt_transcription_ms,
                agent_runtime_ms=llm_latency_ms,
                pure_llm_ms=pure_llm_latency_ms,
                tool_execution_ms=tool_latency_ms,
                tts_synthesis_ms=tts_synthesis_ms,
                total_turn_ms=total_duration_ms,
            )
            logger.info("Voice turn metrics: %s", metrics.to_compact_log())
            raise

    async def execute_turn_from_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        speak_response: bool = True,
        conversation_id: Optional[str] = None,
    ) -> VoiceTurnResult:
        """Execute a full voice turn from provided audio samples.

        Args:
            audio_data: Numpy float32 audio samples.
            sample_rate: Sample rate of the audio.
            speak_response: Whether to synthesize and play the final response.
            conversation_id: Optional conversation ID to persist or resume context.

        Returns:
            VoiceTurnResult: Turn details and latency breakdown.
        """
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn_start = time.perf_counter()
        async with self._turn_lifecycle() as ctx:
            return await self._execute_turn_from_audio_core(
                audio_data=audio_data,
                sample_rate=sample_rate,
                speak_response=speak_response,
                conversation_id=conversation_id,
                lifecycle_context=ctx,
                turn_id=turn_id,
                turn_type="audio",
                turn_start=turn_start,
                audio_capture_ms=None,
                ttfa_reference_start=turn_start,
            )

    async def execute_push_to_talk_turn(
        self,
        stop_event: threading.Event,
        max_duration_sec: float = 15.0,
        speak_response: bool = True,
        conversation_id: Optional[str] = None,
    ) -> VoiceTurnResult:
        """Record audio via push-to-talk and execute voice turn."""
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn_start = time.perf_counter()
        async with self._turn_lifecycle() as ctx:
            await self._wait_pre_recording_cooldown()
            t_cap_start = time.perf_counter()
            try:
                audio = await asyncio.to_thread(
                    self.recorder.record_push_to_talk,
                    stop_event,
                    max_duration_sec,
                )
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
            except asyncio.CancelledError:
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
                total_ms = (time.perf_counter() - turn_start) * 1000.0
                metrics = VoiceTurnMetrics(
                    turn_id=turn_id,
                    turn_type="push_to_talk",
                    success=False,
                    error="Voice turn cancelled",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=0.0,
                    total_turn_ms=total_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                raise
            except Exception as exc:
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
                total_ms = (time.perf_counter() - turn_start) * 1000.0
                logger.error("Recording error in push-to-talk: %s", exc)
                metrics = VoiceTurnMetrics(
                    turn_id=turn_id,
                    turn_type="push_to_talk",
                    success=False,
                    error=f"Microphone recording error: {exc}",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=0.0,
                    total_turn_ms=total_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query="",
                    final_response="",
                    conversation_id=conversation_id or self.conversation_id,
                    total_latency_ms=total_ms,
                    success=False,
                    error=f"Microphone recording error: {exc}",
                    metrics=metrics,
                )

            return await self._execute_turn_from_audio_core(
                audio_data=audio,
                sample_rate=self.recorder.sample_rate,
                speak_response=speak_response,
                conversation_id=conversation_id,
                lifecycle_context=ctx,
                turn_id=turn_id,
                turn_type="push_to_talk",
                turn_start=turn_start,
                audio_capture_ms=audio_capture_ms,
                ttfa_reference_start=t_cap_end,
            )

    async def execute_voice_turn(
        self,
        silence_duration_sec: float = 1.2,
        max_duration_sec: float = 15.0,
        energy_threshold: float = 0.015,
        speak_response: bool = True,
        conversation_id: Optional[str] = None,
    ) -> VoiceTurnResult:
        """Record audio with automatic silence endpointing and execute voice turn."""
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn_start = time.perf_counter()
        async with self._turn_lifecycle() as ctx:
            await self._wait_pre_recording_cooldown()
            t_cap_start = time.perf_counter()
            try:
                audio = await asyncio.to_thread(
                    self.recorder.record_until_silence,
                    silence_duration_sec,
                    max_duration_sec,
                    energy_threshold,
                )
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
            except asyncio.CancelledError:
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
                total_ms = (time.perf_counter() - turn_start) * 1000.0
                metrics = VoiceTurnMetrics(
                    turn_id=turn_id,
                    turn_type="voice",
                    success=False,
                    error="Voice turn cancelled",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=0.0,
                    total_turn_ms=total_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                raise
            except Exception as exc:
                t_cap_end = time.perf_counter()
                audio_capture_ms = (t_cap_end - t_cap_start) * 1000.0
                total_ms = (time.perf_counter() - turn_start) * 1000.0
                logger.error("Recording error in silence detection: %s", exc)
                metrics = VoiceTurnMetrics(
                    turn_id=turn_id,
                    turn_type="voice",
                    success=False,
                    error=f"Microphone recording error: {exc}",
                    audio_capture_ms=audio_capture_ms,
                    user_audio_duration_sec=0.0,
                    total_turn_ms=total_ms,
                )
                logger.info("Voice turn metrics: %s", metrics.to_compact_log())
                return VoiceTurnResult(
                    user_query="",
                    final_response="",
                    conversation_id=conversation_id or self.conversation_id,
                    total_latency_ms=total_ms,
                    success=False,
                    error=f"Microphone recording error: {exc}",
                    metrics=metrics,
                )

            return await self._execute_turn_from_audio_core(
                audio_data=audio,
                sample_rate=self.recorder.sample_rate,
                speak_response=speak_response,
                conversation_id=conversation_id,
                lifecycle_context=ctx,
                turn_id=turn_id,
                turn_type="voice",
                turn_start=turn_start,
                audio_capture_ms=audio_capture_ms,
                ttfa_reference_start=t_cap_end,
            )

    async def run_hands_free_loop(
        self,
        conversation_id: Optional[str] = None,
        stop_event: Optional[Any] = None,
        max_turns: Optional[int] = None,
        silence_duration_sec: float = 1.2,
        max_duration_sec: float = 15.0,
        energy_threshold: float = 0.015,
        speak_response: bool = True,
        backoff_on_no_speech_sec: float = 0.05,
        max_consecutive_device_errors: int = 3,
        on_turn_completed: Optional[Callable[[VoiceTurnResult], Any]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
    ) -> list[VoiceTurnResult]:
        """Execute a continuous hands-free voice loop.

        Repeatedly listens for user utterances, performs local Whisper transcription,
        dispatches to AgentRuntime, synthesizes and plays back speech with Pocket TTS,
        enforces acoustic cooldown, and automatically returns to listening without
        requiring any keypress between turns.

        Preserves conversation_id across turns in the session.
        Handles reflection guard, approvals, device failures, and clean shutdown.

        Args:
            conversation_id: Persistent conversation ID reused across all turns in session.
            stop_event: Optional threading.Event or asyncio.Event to trigger graceful stop.
            max_turns: Optional turn cap for testing or bounded sessions.
            silence_duration_sec: Silence duration after speech before concluding turn.
            max_duration_sec: Maximum recording duration per turn.
            energy_threshold: RMS amplitude threshold for silence endpointing.
            speak_response: Whether to synthesize and speak the assistant's reply.
            backoff_on_no_speech_sec: Pause interval when no speech is heard to prevent busy-looping.
            max_consecutive_device_errors: Max consecutive microphone errors before stopping.
            on_turn_completed: Optional callback invoked after each completed turn.
            on_state_change: Optional callback invoked on lifecycle state changes.

        Returns:
            list[VoiceTurnResult]: Chronological sequence of all turn results in this loop.
        """
        active_cid = conversation_id or self.conversation_id or f"conv_voice_{uuid.uuid4().hex[:8]}"
        self.conversation_id = active_cid

        if not self.recorder.is_input_available():
            logger.error("No microphone or audio input devices detected at start of hands-free loop.")
            if on_state_change:
                on_state_change("device_unavailable")
            return []

        consecutive_device_errors = 0
        turns_completed: list[VoiceTurnResult] = []

        try:
            while True:
                # 1. Check external stop event
                if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                    logger.info("Hands-free voice loop received stop signal.")
                    if on_state_change:
                        on_state_change("stopped")
                    break

                # 2. Check turn count cap
                if max_turns is not None and len(turns_completed) >= max_turns:
                    logger.info("Hands-free voice loop reached maximum turn count (%d).", max_turns)
                    if on_state_change:
                        on_state_change("max_turns_reached")
                    break

                # 3. Notify listening state
                if on_state_change:
                    on_state_change("listening")

                # 4. Execute turn
                try:
                    turn_res = await self.execute_voice_turn(
                        silence_duration_sec=silence_duration_sec,
                        max_duration_sec=max_duration_sec,
                        energy_threshold=energy_threshold,
                        speak_response=speak_response,
                        conversation_id=active_cid,
                    )
                except asyncio.CancelledError:
                    logger.info("Hands-free voice loop cancelled during turn.")
                    raise
                except Exception as exc:
                    consecutive_device_errors += 1
                    logger.warning(
                        "Unexpected error during hands-free turn execution (consecutive=%d/%d): %s",
                        consecutive_device_errors,
                        max_consecutive_device_errors,
                        exc,
                    )
                    if consecutive_device_errors >= max_consecutive_device_errors:
                        logger.error(
                            "Persistent failure in hands-free loop (%d consecutive errors). Stopping.",
                            consecutive_device_errors,
                        )
                        if on_state_change:
                            on_state_change("persistent_failure")
                        break
                    await asyncio.sleep(0.5)
                    continue

                # 5. Device failure check
                if not turn_res.success and turn_res.error and (
                    "Microphone recording error" in turn_res.error or "No audio input device" in turn_res.error
                ):
                    consecutive_device_errors += 1
                    logger.warning(
                        "Microphone device failure (consecutive=%d/%d): %s",
                        consecutive_device_errors,
                        max_consecutive_device_errors,
                        turn_res.error,
                    )
                    turns_completed.append(turn_res)
                    if on_turn_completed:
                        cb_res = on_turn_completed(turn_res)
                        if asyncio.iscoroutine(cb_res):
                            await cb_res
                    if consecutive_device_errors >= max_consecutive_device_errors:
                        logger.error(
                            "Persistent microphone device failure (%d consecutive errors). Exiting hands-free loop.",
                            consecutive_device_errors,
                        )
                        if on_state_change:
                            on_state_change("device_error")
                        break
                    await asyncio.sleep(0.5)
                    continue

                # Valid recording produced (speech, silence, or reflection) -> reset error counter
                consecutive_device_errors = 0
                turns_completed.append(turn_res)

                # Invoke turn completed callback
                if on_turn_completed:
                    cb_res = on_turn_completed(turn_res)
                    if asyncio.iscoroutine(cb_res):
                        await cb_res

                # 6. Handle no speech / silence outcome
                if turn_res.error == "No speech detected" or not turn_res.user_query.strip():
                    if on_state_change:
                        on_state_change("no_speech")
                    if backoff_on_no_speech_sec > 0:
                        await asyncio.sleep(backoff_on_no_speech_sec)
                    continue

                # 7. Handle reflection guard suppression
                if turn_res.error == "Transcript suppressed by reflection guard":
                    if on_state_change:
                        on_state_change("reflection_suppressed")
                    continue

                # 8. Check for user voice exit command
                clean_query = turn_res.user_query.strip().lower().rstrip(".!?,")
                if clean_query in {"exit", "quit", "stop listening", "goodbye", "shut down", "bye"}:
                    logger.info("User voice exit command detected: '%s'. Stopping hands-free loop.", clean_query)
                    if on_state_change:
                        on_state_change("exiting")
                    break

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Hands-free voice loop interrupted or cancelled.")
            raise
        finally:
            self.tts_engine.stop_playback()

        return turns_completed

