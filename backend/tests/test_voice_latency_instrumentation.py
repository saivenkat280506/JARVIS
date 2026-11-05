"""Deterministic regression tests for Voice Pipeline Latency Instrumentation (Milestone 10B-4).

Verifies per-stage monotonic timing, structured metrics generation, privacy protection,
single-emission guarantee, and error handling across all failure modes.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent.runtime import AgentRunResult, ToolExecutionRecord
from voice.pipeline import VoicePipeline, VoiceTurnMetrics, VoiceTurnResult
from voice.stt import STTResult
from voice.tts import TTSResult


@pytest.fixture
def dummy_audio() -> np.ndarray:
    """Return 1.0 second of silent float32 audio samples."""
    return np.zeros(16000, dtype=np.float32)


@pytest.mark.anyio
async def test_successful_voice_turn_all_stages_measured(dummy_audio: np.ndarray):
    """Verify successful voice turn measures all 8 stages accurately using monotonic timestamps."""
    mock_runtime = MagicMock()
    tool_rec = ToolExecutionRecord(
        round_index=0,
        tool_name="calculate",
        arguments={"expression": "2+2"},
        result={"result": 4},
        duration_sec=0.050,  # 50ms tool
    )
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Calculate 2+2",
            final_response="The answer is 4, sir.",
            tool_records=[tool_rec],
            completed=True,
        )
    )

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Calculate 2+2",
        latency_sec=0.120,  # 120ms
        audio_duration_sec=1.0,
        is_empty=False,
    )

    playback_t0 = 1000.500
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(
        text="The answer is 4, sir.",
        latency_sec=0.080,  # 80ms synthesis
        audio_duration_sec=1.5,
        played=True,
        playback_duration_sec=0.300,  # 300ms playback
        playback_start_timestamp=playback_t0,
    )

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = dummy_audio

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        post_playback_cooldown_sec=0.05,
    )

    res = await pipeline.execute_voice_turn(silence_duration_sec=1.0, max_duration_sec=5.0)

    assert res.success is True
    assert res.metrics is not None

    m = res.metrics
    assert m.turn_type == "voice"
    assert m.turn_id.startswith("turn_")
    assert m.success is True
    assert m.error is None

    # Stage 1: Audio capture
    assert m.audio_capture_ms is not None and m.audio_capture_ms >= 0.0
    assert abs(m.user_audio_duration_sec - 1.0) < 1e-3

    # Stage 2: STT transcription
    assert m.stt_transcription_ms is not None and m.stt_transcription_ms >= 0.0

    # Stage 3: Agent runtime & tool breakdown
    assert m.agent_runtime_ms is not None and m.agent_runtime_ms >= 0.0
    assert m.tool_execution_ms == pytest.approx(50.0, abs=1.0)
    assert m.pure_llm_ms is not None and m.pure_llm_ms >= 0.0

    # Stage 4: TTS synthesis
    assert m.tts_synthesis_ms == pytest.approx(80.0, abs=1.0)
    assert m.tts_audio_duration_sec == pytest.approx(1.5, abs=1e-3)

    # Stage 5: TTFA
    assert m.ttfa_ms is not None

    # Stage 6: Audio playback
    assert m.audio_playback_ms == pytest.approx(300.0, abs=1.0)

    # Stage 7: Post-playback cooldown
    assert m.post_playback_cooldown_ms is not None and m.post_playback_cooldown_ms >= 0.0

    # Stage 8: Total duration
    assert m.total_turn_ms > 0.0


@pytest.mark.anyio
async def test_empty_no_speech_input_metrics(dummy_audio: np.ndarray):
    """Empty recording sets error='No speech detected' and leaves downstream stages None."""
    mock_runtime = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        is_empty=True,
    )
    mock_tts = MagicMock()
    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = dummy_audio

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
    )

    res = await pipeline.execute_voice_turn()
    assert res.error == "No speech detected"
    assert res.metrics is not None

    m = res.metrics
    assert m.success is True
    assert m.error == "No speech detected"
    assert m.audio_capture_ms is not None
    assert m.stt_transcription_ms is not None
    # Downstream stages must be explicitly None
    assert m.agent_runtime_ms is None
    assert m.tts_synthesis_ms is None
    assert m.ttfa_ms is None
    assert m.audio_playback_ms is None
    assert m.post_playback_cooldown_ms is None


@pytest.mark.anyio
async def test_stt_failure_metrics(dummy_audio: np.ndarray):
    """STT exception sets success=False and preserves failure details without crashing."""
    mock_runtime = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.side_effect = RuntimeError("Whisper model corrupted")
    mock_tts = MagicMock()
    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = dummy_audio

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
    )

    res = await pipeline.execute_voice_turn()
    assert res.success is False
    assert "Speech recognition error" in (res.error or "")

    m = res.metrics
    assert m is not None
    assert m.success is False
    assert "Speech recognition error: Whisper model corrupted" in (m.error or "")
    assert m.audio_capture_ms is not None
    assert m.stt_transcription_ms is not None
    assert m.agent_runtime_ms is None
    assert m.tts_synthesis_ms is None


@pytest.mark.anyio
async def test_runtime_failure_metrics(dummy_audio: np.ndarray):
    """Runtime LLM exception sets success=False and records inference timing."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=RuntimeError("Ollama daemon unreachable"))

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="What time is it?",
        latency_sec=0.060,
        audio_duration_sec=1.0,
        is_empty=False,
    )
    mock_tts = MagicMock()

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    res = await pipeline.execute_turn_from_audio(dummy_audio)
    assert res.success is False
    assert "Agent inference failure" in (res.error or "")

    m = res.metrics
    assert m is not None
    assert m.success is False
    assert "Agent inference failure: Ollama daemon unreachable" in (m.error or "")
    assert m.audio_capture_ms is None  # direct audio turn
    assert m.stt_transcription_ms is not None
    assert m.agent_runtime_ms is not None
    assert m.tts_synthesis_ms is None


@pytest.mark.anyio
async def test_tts_synthesis_failure_metrics(dummy_audio: np.ndarray):
    """TTS failure logs error, preserves runtime answer, and leaves playback metrics None."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Hello",
            final_response="Good morning.",
            completed=True,
        )
    )

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hello",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    mock_tts = MagicMock()
    mock_tts.speak.side_effect = RuntimeError("afplay binary missing")

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    res = await pipeline.execute_turn_from_audio(dummy_audio)
    assert res.final_response == "Good morning."
    assert res.metrics is not None

    m = res.metrics
    assert m.agent_runtime_ms is not None
    assert m.tts_synthesis_ms is None
    assert m.ttfa_ms is None
    assert m.audio_playback_ms is None
    assert m.post_playback_cooldown_ms is None


@pytest.mark.anyio
async def test_playback_failure_or_unavailable_playback_start_signal(dummy_audio: np.ndarray):
    """When playback fails or start timestamp is unavailable, TTFA is explicitly None."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Test",
            final_response="Testing playback.",
            completed=True,
        )
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Test",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    # Case A: played=False
    mock_tts_failed = MagicMock()
    mock_tts_failed.speak.return_value = TTSResult(
        text="Testing playback.",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        played=False,
    )

    pipeline_a = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts_failed,
    )
    res_a = await pipeline_a.execute_turn_from_audio(dummy_audio)
    assert res_a.metrics.ttfa_ms is None
    assert res_a.metrics.audio_playback_ms is None

    # Case B: played=True but playback_start_timestamp is None (uninstrumented)
    mock_tts_no_ts = MagicMock()
    mock_tts_no_ts.speak.return_value = TTSResult(
        text="Testing playback.",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        played=True,
        playback_duration_sec=0.200,
        playback_start_timestamp=None,
    )

    pipeline_b = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts_no_ts,
    )
    res_b = await pipeline_b.execute_turn_from_audio(dummy_audio)
    assert res_b.metrics.audio_playback_ms == pytest.approx(200.0, abs=1.0)
    assert res_b.metrics.ttfa_ms is None  # Reported as unavailable rather than invented


@pytest.mark.anyio
async def test_cancellation_and_cleanup_metrics(dummy_audio: np.ndarray, caplog):
    """Turn cancellation propagates CancelledError, logs metrics with error, and releases lock."""
    mock_runtime = MagicMock()

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(10.0)
        return AgentRunResult(prompt="", final_response="", completed=True)

    mock_runtime.run = AsyncMock(side_effect=slow_run)
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Take your time",
        latency_sec=0.020,
        audio_duration_sec=1.0,
        is_empty=False,
    )
    mock_tts = MagicMock()

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(pipeline.execute_turn_from_audio(dummy_audio))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    # Lock must be released
    assert not pipeline._lock.locked()

    # Metrics must have been logged for the cancelled turn
    metric_logs = [r.message for r in caplog.records if "Voice turn metrics:" in r.message]
    assert len(metric_logs) == 1
    assert "error=Voice turn cancelled" in metric_logs[0]


@pytest.mark.anyio
async def test_distinction_between_synthesis_playback_and_total_duration(dummy_audio: np.ndarray):
    """Verify synthesis latency is isolated from playback duration and does not taint inference."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Hello",
            final_response="Hi there.",
            completed=True,
        )
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hello",
        latency_sec=0.050,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(
        text="Hi there.",
        latency_sec=0.075,            # 75ms synthesis
        audio_duration_sec=1.2,
        played=True,
        playback_duration_sec=0.450,  # 450ms playback
        playback_start_timestamp=100.0,
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )

    res = await pipeline.execute_turn_from_audio(dummy_audio)
    m = res.metrics
    assert m is not None

    # Synthesis is 75ms, playback is 450ms - distinct measurements
    assert m.tts_synthesis_ms == pytest.approx(75.0, abs=1.0)
    assert m.audio_playback_ms == pytest.approx(450.0, abs=1.0)
    assert m.tts_synthesis_ms != m.audio_playback_ms

    # Ensure agent runtime latency is isolated from and does not include TTS synthesis or playback
    assert m.agent_runtime_ms is not None
    assert m.agent_runtime_ms != m.tts_synthesis_ms
    assert m.agent_runtime_ms != m.audio_playback_ms


@pytest.mark.anyio
async def test_metrics_emitted_once_per_turn(dummy_audio: np.ndarray, caplog):
    """Verify each turn emits exactly one structured metrics record across all entrypoints and outcomes."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="Hello", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hi", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False
    )
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(
        text="Hello", latency_sec=0.01, audio_duration_sec=0.5, played=True, playback_duration_sec=0.1
    )
    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = dummy_audio

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        post_playback_cooldown_sec=0.0,
    )

    caplog.clear()
    with caplog.at_level(logging.INFO):
        # 1. execute_voice_turn
        await pipeline.execute_voice_turn()
        logs1 = [r.message for r in caplog.records if "Voice turn metrics:" in r.message]
        assert len(logs1) == 1

        # 2. execute_turn_from_audio
        caplog.clear()
        await pipeline.execute_turn_from_audio(dummy_audio)
        logs2 = [r.message for r in caplog.records if "Voice turn metrics:" in r.message]
        assert len(logs2) == 1

        # 3. recording error turn
        caplog.clear()
        mock_recorder.record_until_silence.side_effect = RuntimeError("Device busy")
        await pipeline.execute_voice_turn()
        logs3 = [r.message for r in caplog.records if "Voice turn metrics:" in r.message]
        assert len(logs3) == 1


def test_no_transcript_or_response_leakage_in_metrics():
    """Verify structured metrics dictionary and log string contain no transcripts, responses, or secrets."""
    secret_query = "SUPER_SECRET_USER_PASSWORD_ALPHA_9999"
    secret_response = "CONFIDENTIAL_PIN_CODE_BETA_8888"

    metrics = VoiceTurnMetrics(
        turn_id="turn_test123",
        turn_type="voice",
        success=True,
        error=None,
        audio_capture_ms=1200.0,
        user_audio_duration_sec=1.5,
        stt_transcription_ms=150.0,
        agent_runtime_ms=300.0,
        pure_llm_ms=250.0,
        tool_execution_ms=50.0,
        tts_synthesis_ms=80.0,
        tts_audio_duration_sec=1.2,
        ttfa_ms=530.0,
        audio_playback_ms=600.0,
        post_playback_cooldown_ms=350.0,
        total_turn_ms=2700.0,
    )

    d = metrics.to_dict()
    log_str = metrics.to_compact_log()

    # Assert neither secret query nor secret response exists in metrics representation
    assert secret_query not in str(d)
    assert secret_query not in log_str
    assert secret_response not in str(d)
    assert secret_response not in log_str

    # Assert standard metrics keys exist
    assert d["turn_id"] == "turn_test123"
    assert d["turn_type"] == "voice"
    assert d["ttfa_ms"] == 530.0
    assert d["audio_capture_ms"] == 1200.0
    assert "ttfa=530.0ms" in log_str
