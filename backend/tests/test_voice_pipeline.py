"""Unit tests for Voice Pipeline orchestration, audio coordination, and latency tracking."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from agent.runtime import AgentRunResult, ToolExecutionRecord
from tools.music import MusicState, default_music_controller
from voice.pipeline import VoicePipeline, VoiceTurnResult
from voice.stt import STTResult
from voice.tts import TTSResult


@pytest.mark.anyio
async def test_voice_pipeline_successful_turn():
    mock_runtime = MagicMock()
    tool_rec = ToolExecutionRecord(
        round_index=1,
        tool_name="calculate",
        arguments={"expression": "2+2"},
        result={"result": 4},
        duration_sec=0.01,
    )
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="What is 2 plus 2?",
            final_response="The answer is 4, sir.",
            messages=[],
            tool_records=[tool_rec],
            rounds_used=1,
            completed=True,
        )
    )


    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="What is 2 plus 2?",
        latency_sec=0.15,
        audio_duration_sec=1.5,
        is_empty=False,
    )

    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(
        text="The answer is 4, sir.",
        latency_sec=0.35,
        audio_duration_sec=2.0,
        played=True,
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    result: VoiceTurnResult = await pipeline.execute_turn_from_audio(dummy_audio, sample_rate=16000)

    assert result.user_query == "What is 2 plus 2?"
    assert result.final_response == "The answer is 4, sir."
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "calculate"
    assert result.stt_latency_ms == pytest.approx(150.0, 1.0)
    assert result.tts_latency_ms == pytest.approx(350.0, 1.0)
    assert result.total_latency_ms > 0
    assert result.success is True

    mock_stt.transcribe_audio.assert_called_once()
    mock_runtime.run.assert_called_once_with("What is 2 plus 2?")
    mock_tts.speak.assert_called_once_with("The answer is 4, sir.")


@pytest.mark.anyio
async def test_voice_pipeline_silence_no_speech():
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock()

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="",
        latency_sec=0.05,
        audio_duration_sec=1.0,
        is_empty=True,
    )

    mock_tts = MagicMock()
    mock_tts.speak = MagicMock()

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    assert result.user_query == ""
    assert result.error == "No speech detected"
    # Verify LLM and TTS were NEVER called on silence
    mock_runtime.run.assert_not_called()
    mock_tts.speak.assert_not_called()


@pytest.mark.anyio
async def test_voice_pipeline_music_coordination():
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Hello JARVIS",
            final_response="Music paused and resumed cleanly.",
            messages=[],
            tool_records=[],
            rounds_used=1,
            completed=True,
        )
    )

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hello JARVIS",
        latency_sec=0.1,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(
        text="Music paused and resumed cleanly.",
        latency_sec=0.2,
        audio_duration_sec=1.5,
        played=True,
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    # Simulate music being currently active
    with patch.object(default_music_controller, "get_state", side_effect=[
        MusicState.PLAYING,  # Before TTS: detected playing
        MusicState.PAUSED,   # After TTS: paused state, so should resume
    ]), \
    patch.object(default_music_controller, "pause") as mock_pause, \
    patch.object(default_music_controller, "resume") as mock_resume:

        dummy_audio = np.ones(16000, dtype=np.float32)
        result = await pipeline.execute_turn_from_audio(dummy_audio)

        assert result.success is True
        mock_pause.assert_called_once()
        mock_resume.assert_called_once()


@pytest.mark.anyio
async def test_voice_pipeline_runtime_failure_handling():
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=RuntimeError("Ollama inference timeout"))

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hello",
        latency_sec=0.1,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=MagicMock(),
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    assert result.success is False
    assert "inference failure" in (result.error or "")


@pytest.mark.anyio
async def test_voice_pipeline_recording_failure_handling():
    mock_recorder = MagicMock()
    mock_recorder.record_push_to_talk.side_effect = RuntimeError("Microphone disconnected")

    pipeline = VoicePipeline(
        runtime=MagicMock(),
        recorder=mock_recorder,
    )

    import threading
    stop_event = threading.Event()
    result = await pipeline.execute_push_to_talk_turn(stop_event)

    assert result.success is False
    assert "Microphone recording error" in (result.error or "")


@pytest.mark.anyio
async def test_voice_pipeline_tts_failure_does_not_crash():
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Hello",
            final_response="Good day, sir.",
            messages=[],
            tool_records=[],
            rounds_used=1,
            completed=True,
        )
    )

    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Hello",
        latency_sec=0.1,
        audio_duration_sec=1.0,
        is_empty=False,
    )

    mock_tts = MagicMock()
    mock_tts.speak.side_effect = RuntimeError("Audio device busy")

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    # Pipeline successfully delivered final response even though TTS speaker failed
    assert result.success is True
    assert result.final_response == "Good day, sir."


# ==============================================================================
# Milestone 10B-1 Regression Tests: Audio Lifecycle & Concurrency Safety
# ==============================================================================

@pytest.mark.anyio
async def test_music_stays_paused_throughout_entire_turn():
    """Prove that music is paused before recording and remains paused across
    STT transcription, agent inference, and TTS playback, resuming only after turn ends."""
    import threading

    events_log = []

    mock_recorder = MagicMock()
    def fake_record(*args, **kwargs):
        events_log.append("record")
        return np.ones(16000, dtype=np.float32)
    mock_recorder.record_push_to_talk.side_effect = fake_record
    mock_recorder.sample_rate = 16000

    mock_stt = MagicMock()
    def fake_stt(*args, **kwargs):
        events_log.append("stt")
        return STTResult(text="Turn query", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)
    mock_stt.transcribe_audio.side_effect = fake_stt

    mock_runtime = MagicMock()
    async def fake_run(*args, **kwargs):
        events_log.append("llm")
        return AgentRunResult(prompt="Turn query", final_response="Turn reply", completed=True)
    mock_runtime.run = AsyncMock(side_effect=fake_run)

    mock_tts = MagicMock()
    def fake_tts(*args, **kwargs):
        events_log.append("tts")
        return TTSResult(text="Turn reply", latency_sec=0.1, audio_duration_sec=1.0, played=True)
    mock_tts.speak.side_effect = fake_tts

    # Mock music controller to track order of pause/resume
    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,  # Pre-turn state
        MusicState.PAUSED,   # Post-turn state before resume
    ]

    def on_pause():
        events_log.append("pause")
        return {"success": True}
    def on_resume():
        events_log.append("resume")
        return {"success": True}

    mock_music.pause.side_effect = on_pause
    mock_music.resume.side_effect = on_resume

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        music_controller=mock_music,
    )

    stop_event = threading.Event()
    result = await pipeline.execute_push_to_talk_turn(stop_event)

    assert result.success is True
    # Verify exact ordering: pause occurs before record, and resume occurs strictly after tts
    assert events_log == ["pause", "record", "stt", "llm", "tts", "resume"]
    mock_music.pause.assert_called_once()
    mock_music.resume.assert_called_once()
    assert not pipeline._lock.locked()


@pytest.mark.anyio
@pytest.mark.parametrize("initial_state", [MusicState.STOPPED, MusicState.PAUSED])
async def test_music_not_started_if_previously_stopped_or_paused(initial_state):
    """Prove that music that was STOPPED or PAUSED before the turn is not resumed or started."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="Hello", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Hi", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Hello", latency_sec=0.05, audio_duration_sec=1.0, played=True)

    mock_music = MagicMock()
    mock_music.get_state.return_value = initial_state

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        music_controller=mock_music,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    assert result.success is True
    # Ensure neither pause nor resume was called
    mock_music.pause.assert_not_called()
    mock_music.resume.assert_not_called()
    assert not pipeline._lock.locked()


@pytest.mark.anyio
@pytest.mark.parametrize("tool_name, resulting_state", [
    ("stop_music", MusicState.STOPPED),
    ("pause_music", MusicState.PAUSED),
])
async def test_music_not_resumed_if_tool_stopped_or_paused_music(tool_name, resulting_state):
    """If user asks JARVIS to stop or pause music, the turn must NOT resume music."""
    mock_runtime = MagicMock()
    tool_rec = ToolExecutionRecord(
        round_index=1,
        tool_name=tool_name,
        arguments={},
        result={"success": True},
        duration_sec=0.01,
    )
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Stop music",
            final_response="Music stopped, sir.",
            tool_records=[tool_rec],
            completed=True,
        )
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Stop music", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Music stopped, sir.", latency_sec=0.05, audio_duration_sec=1.0, played=True)

    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,  # Before turn: playing
        resulting_state,     # After turn: state reflecting tool action
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        music_controller=mock_music,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    assert result.success is True
    mock_music.pause.assert_called_once()
    mock_music.resume.assert_not_called()
    assert not pipeline._lock.locked()


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["recording", "transcription", "inference", "tts"])
async def test_music_restoration_attempted_after_failure_at_any_stage(stage):
    """Verify that if music was active before a turn, music restoration is attempted
    even if failure occurs during recording, transcription, LLM inference, or TTS."""
    import threading

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    if stage == "recording":
        mock_recorder.record_push_to_talk.side_effect = RuntimeError("Microphone disconnected")
    else:
        mock_recorder.record_push_to_talk.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock()
    if stage == "transcription":
        mock_stt.transcribe_audio.side_effect = RuntimeError("Whisper transcription failure")
    else:
        mock_stt.transcribe_audio.return_value = STTResult(text="Query", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)

    mock_runtime = MagicMock()
    if stage == "inference":
        mock_runtime.run = AsyncMock(side_effect=RuntimeError("Ollama inference crash"))
    else:
        mock_runtime.run = AsyncMock(return_value=AgentRunResult(prompt="Query", final_response="Answer", completed=True))

    mock_tts = MagicMock()
    if stage == "tts":
        mock_tts.speak.side_effect = RuntimeError("Audio device busy")
    else:
        mock_tts.speak.return_value = TTSResult(text="Answer", latency_sec=0.05, audio_duration_sec=1.0, played=True)

    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,  # Before turn
        MusicState.PAUSED,   # After turn / cleanup
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        music_controller=mock_music,
    )

    stop_event = threading.Event()
    result = await pipeline.execute_push_to_talk_turn(stop_event)

    # In TTS failure, pipeline still completes final response gracefully; in earlier failures, success=False
    if stage == "tts":
        assert result.success is True
    else:
        assert result.success is False

    # In ALL cases, pause was called and resume was attempted
    mock_music.pause.assert_called_once()
    mock_music.resume.assert_called_once()
    # And lock is cleanly released
    assert not pipeline._lock.locked()


@pytest.mark.anyio
async def test_concurrent_turns_serialized_no_overlap():
    """Verify that concurrent calls to the same pipeline do not overlap
    microphone capture or turn processing."""
    import threading

    active_in_pipeline = 0
    max_concurrent = 0
    turn_1_entered = asyncio.Event()
    turn_1_can_proceed = asyncio.Event()

    mock_runtime = MagicMock()
    mock_stt = MagicMock()
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Hi", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000

    def fake_record(*args, **kwargs):
        nonlocal active_in_pipeline, max_concurrent
        active_in_pipeline += 1
        max_concurrent = max(max_concurrent, active_in_pipeline)
        return np.ones(16000, dtype=np.float32)

    mock_recorder.record_push_to_talk.side_effect = fake_record

    async def fake_run(query, **kwargs):
        nonlocal active_in_pipeline, max_concurrent
        # During turn 1's inference, signal that turn 1 is active and wait
        if query == "turn1":
            turn_1_entered.set()
            await turn_1_can_proceed.wait()
        return AgentRunResult(prompt=query, final_response=f"Reply to {query}", completed=True)

    mock_runtime.run = AsyncMock(side_effect=fake_run)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
    )

    stop_event = threading.Event()
    dummy_audio_1 = np.ones(16000, dtype=np.float32)
    mock_stt.transcribe_audio.side_effect = [
        STTResult(text="turn1", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="turn2", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False),
    ]

    async def run_turn_1():
        nonlocal active_in_pipeline
        res = await pipeline.execute_turn_from_audio(dummy_audio_1)
        active_in_pipeline -= 1
        return res

    async def run_turn_2():
        nonlocal active_in_pipeline
        res = await pipeline.execute_push_to_talk_turn(stop_event)
        active_in_pipeline -= 1
        return res

    # Launch turn 1
    t1_task = asyncio.create_task(run_turn_1())
    # Wait until turn 1 is actively inside its turn (in inference)
    await turn_1_entered.wait()
    assert pipeline._lock.locked() is True

    # Launch turn 2 while turn 1 is still running
    t2_task = asyncio.create_task(run_turn_2())
    # Give turn 2 task a tick to attempt entering
    await asyncio.sleep(0.01)

    # Prove turn 2 did NOT enter recording while turn 1 is active
    assert mock_recorder.record_push_to_talk.call_count == 0

    # Now allow turn 1 to complete
    turn_1_can_proceed.set()

    res1, res2 = await asyncio.gather(t1_task, t2_task)

    assert res1.user_query == "turn1"
    assert res2.user_query == "turn2"
    assert mock_recorder.record_push_to_talk.call_count == 1
    assert max_concurrent <= 1
    assert not pipeline._lock.locked()


@pytest.mark.anyio
async def test_locks_released_on_unhandled_exception():
    """Verify that unexpected exceptions release the lock so subsequent turns can proceed."""
    import threading

    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(return_value=AgentRunResult(prompt="", final_response="", completed=True))
    mock_recorder = MagicMock()
    mock_recorder.record_push_to_talk.side_effect = KeyboardInterrupt("Simulated interrupt")

    pipeline = VoicePipeline(runtime=mock_runtime, recorder=mock_recorder)

    stop_event = threading.Event()

    with pytest.raises(KeyboardInterrupt):
        await pipeline.execute_push_to_talk_turn(stop_event)

    # Lock must be released
    assert not pipeline._lock.locked()

    # Subsequent turn must not deadlock
    mock_recorder.record_push_to_talk.side_effect = None
    mock_recorder.record_push_to_talk.return_value = np.zeros(16000, dtype=np.float32)
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="", latency_sec=0.01, audio_duration_sec=1.0, is_empty=True)
    pipeline.stt_engine = mock_stt

    res = await pipeline.execute_push_to_talk_turn(stop_event)
    assert res.error == "No speech detected"
    assert not pipeline._lock.locked()


@pytest.mark.anyio
async def test_locks_and_music_released_on_cancellation():
    """Verify that cancelling an in-flight voice turn releases the lock and restores music."""
    import threading

    turn_started = asyncio.Event()
    thread_can_exit = threading.Event()

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000

    def fake_record(*args, **kwargs):
        turn_started.set()
        thread_can_exit.wait(timeout=2.0)
        return np.ones(16000, dtype=np.float32)

    mock_recorder.record_push_to_talk.side_effect = fake_record

    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,  # Before turn
        MusicState.PAUSED,   # On cancellation cleanup
    ]

    pipeline = VoicePipeline(
        runtime=MagicMock(),
        recorder=mock_recorder,
        music_controller=mock_music,
    )

    stop_event = threading.Event()
    task = asyncio.create_task(pipeline.execute_push_to_talk_turn(stop_event))

    await turn_started.wait()
    assert pipeline._lock.locked() is True
    mock_music.pause.assert_called_once()

    # Cancel the running turn
    task.cancel()
    thread_can_exit.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Lock must be released
    assert not pipeline._lock.locked()
    # Music must be resumed
    mock_music.resume.assert_called_once()


@pytest.mark.anyio
async def test_music_controller_failures_are_graceful():
    """Verify that errors inside music controller pause or resume do not crash the turn."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="All good.", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Hi", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="All good.", latency_sec=0.05, audio_duration_sec=1.0, played=True)

    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,
        MusicState.PAUSED,
    ]
    # Simulate hardware error or process failure on pause and resume
    mock_music.pause.side_effect = OSError("Player process died")
    mock_music.resume.side_effect = OSError("Cannot resume")

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        music_controller=mock_music,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)
    # Turn should succeed without crashing on music exceptions
    result = await pipeline.execute_turn_from_audio(dummy_audio)

    assert result.success is True
    assert result.final_response == "All good."
    assert not pipeline._lock.locked()


@pytest.mark.anyio
async def test_voice_turn_silence_detection_audio_lifecycle():
    """Verify execute_voice_turn coordinates music and lock through silence endpointing."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Auto speech", final_response="Auto reply", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Auto speech", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False)
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Auto reply", latency_sec=0.05, audio_duration_sec=1.0, played=True)

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_music = MagicMock()
    mock_music.get_state.side_effect = [
        MusicState.PLAYING,
        MusicState.PAUSED,
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        music_controller=mock_music,
    )

    result = await pipeline.execute_voice_turn(silence_duration_sec=1.0, max_duration_sec=5.0)

    assert result.success is True
    assert result.final_response == "Auto reply"
    mock_recorder.record_until_silence.assert_called_once_with(1.0, 5.0, 0.015)
    mock_music.pause.assert_called_once()
    mock_music.resume.assert_called_once()
    assert not pipeline._lock.locked()


# ==============================================================================
# Milestone 10B-2 Regression Tests: Voice Self-Capture Protection
# ==============================================================================

@pytest.mark.anyio
async def test_post_playback_cooldown_occurs_after_tts_before_next_recording():
    """Verify that cooldown is applied asynchronously after TTS playback and
    before the next microphone recording begins."""
    events = []

    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="Good day, sir.", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Hi", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)

    mock_tts = MagicMock()
    def fake_speak(text):
        events.append("tts_playback")
        return TTSResult(text=text, latency_sec=0.1, audio_duration_sec=1.0, played=True)
    mock_tts.speak.side_effect = fake_speak

    mock_recorder = MagicMock()
    mock_recorder.sample_rate = 16000
    def fake_record(*args, **kwargs):
        events.append("microphone_recording")
        return np.ones(16000, dtype=np.float32)
    mock_recorder.record_push_to_talk.side_effect = fake_record

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        post_playback_cooldown_sec=0.35,
    )

    import threading
    stop_event = threading.Event()

    sleep_calls = []
    real_sleep = asyncio.sleep

    async def tracking_sleep(duration):
        events.append(f"cooldown_sleep_{duration}")
        sleep_calls.append(duration)
        await real_sleep(0.001)

    with patch("asyncio.sleep", side_effect=tracking_sleep):
        # Turn 1: execute turn with TTS output
        dummy_audio = np.ones(16000, dtype=np.float32)
        res1 = await pipeline.execute_turn_from_audio(dummy_audio)
        assert res1.success is True

        # Turn 2: start push-to-talk recording
        res2 = await pipeline.execute_push_to_talk_turn(stop_event)
        assert res2.success is True

    assert 0.35 in sleep_calls
    # Verify exact event ordering: tts_playback -> cooldown_sleep -> microphone_recording
    assert events.index("tts_playback") < events.index("cooldown_sleep_0.35")
    assert events.index("cooldown_sleep_0.35") < events.index("microphone_recording")


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["speak_false", "playback_unplayed", "no_speech"])
async def test_post_playback_cooldown_skipped_when_no_audio_played(scenario):
    """Verify cooldown is skipped when no audio was spoken by the assistant."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="Answer", completed=True)
    )
    mock_stt = MagicMock()
    mock_tts = MagicMock()

    if scenario == "no_speech":
        mock_stt.transcribe_audio.return_value = STTResult(text="", latency_sec=0.01, audio_duration_sec=1.0, is_empty=True)
    else:
        mock_stt.transcribe_audio.return_value = STTResult(text="Hi", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)

    if scenario == "playback_unplayed":
        mock_tts.speak.return_value = TTSResult(text="Answer", latency_sec=0.01, audio_duration_sec=0.0, played=False)
    else:
        mock_tts.speak.return_value = TTSResult(text="Answer", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.35,
    )

    speak_response = (scenario != "speak_false")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        dummy_audio = np.ones(16000, dtype=np.float32)
        res = await pipeline.execute_turn_from_audio(dummy_audio, speak_response=speak_response)
        assert res.success is True
        mock_sleep.assert_not_called()


@pytest.mark.anyio
async def test_transcript_reflection_guard_suppresses_near_identical_response():
    """Verify that a near-identical transcript echoing the assistant's previous response is suppressed."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Query", final_response="The answer is 42, sir.", completed=True)
    )
    mock_stt = MagicMock()
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="The answer is 42, sir.", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )

    dummy_audio = np.ones(16000, dtype=np.float32)

    # Turn 1: Normal user turn
    mock_stt.transcribe_audio.return_value = STTResult(text="What is the answer?", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)
    res1 = await pipeline.execute_turn_from_audio(dummy_audio)
    assert res1.success is True
    assert res1.final_response == "The answer is 42, sir."
    assert mock_runtime.run.call_count == 1

    # Turn 2: Self-captured reflection echoed back into mic
    mock_stt.transcribe_audio.return_value = STTResult(text="The answer is 42, sir.", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)
    res2 = await pipeline.execute_turn_from_audio(dummy_audio)

    # Must be suppressed by reflection guard
    assert res2.success is True
    assert res2.final_response == ""
    assert res2.error == "Transcript suppressed by reflection guard"
    # Agent runtime must NOT have been called a second time
    assert mock_runtime.run.call_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("reflection_variant", [
    "the answer is 42, sir.",
    "THE ANSWER IS 42 SIR",
    "  The answer   is   42, sir!  ",
    "The answer is 42, sir...",
])
async def test_reflection_guard_normalizes_case_whitespace_and_punctuation(reflection_variant):
    """Verify that harmless case, whitespace, and punctuation differences do not defeat the guard."""
    mock_runtime = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text=reflection_variant, latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "The answer is 42, sir."

    dummy_audio = np.ones(16000, dtype=np.float32)
    res = await pipeline.execute_turn_from_audio(dummy_audio)

    assert res.error == "Transcript suppressed by reflection guard"
    assert res.final_response == ""
    mock_runtime.run.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("command", [
    "Yes",
    "No",
    "Okay",
    "Thank you",
    "Got it",
    "Cancel",
    "Stop",
    "What time is it?",
    "Calculate 2+2",
    "Check my reminders",
])
async def test_reflection_guard_does_not_suppress_acknowledgments_or_ordinary_commands(command):
    """Verify that short acknowledgments and ordinary commands are never suppressed."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt=command, final_response=f"Handled {command}", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text=command, latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Done", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "Would you like me to check your reminders or cancel? Say yes to continue."

    dummy_audio = np.ones(16000, dtype=np.float32)
    res = await pipeline.execute_turn_from_audio(dummy_audio)

    assert res.error is None
    assert res.final_response == f"Handled {command}"
    mock_runtime.run.assert_called_once_with(command)


@pytest.mark.anyio
async def test_legitimate_different_follow_up_reaches_runtime():
    """Verify that a legitimate follow-up sharing vocabulary with the assistant response reaches runtime."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(
            prompt="Tell me more about Philz coffee shop.",
            final_response="Philz Coffee is known for pour-overs.",
            completed=True,
        )
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Tell me more about Philz coffee shop.",
        latency_sec=0.01,
        audio_duration_sec=1.0,
        is_empty=False,
    )
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Details", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "I found three coffee shops nearby: Blue Bottle, Philz, and Starbucks."

    dummy_audio = np.ones(16000, dtype=np.float32)
    res = await pipeline.execute_turn_from_audio(dummy_audio)

    assert res.error is None
    assert res.final_response == "Philz Coffee is known for pour-overs."
    mock_runtime.run.assert_called_once_with("Tell me more about Philz coffee shop.")


@pytest.mark.anyio
async def test_suppressed_text_creates_no_agent_invocation_or_turn():
    """Verify that when reflection is suppressed, AgentRuntime is never invoked,
    no tools are executed, and TTS is not triggered."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock()
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Your reminder has been created for tomorrow at 9:00 AM.",
        latency_sec=0.01,
        audio_duration_sec=1.0,
        is_empty=False,
    )
    mock_tts = MagicMock()

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "Your reminder has been created for tomorrow at 9:00 AM."

    dummy_audio = np.ones(16000, dtype=np.float32)
    res = await pipeline.execute_turn_from_audio(dummy_audio)

    assert res.error == "Transcript suppressed by reflection guard"
    assert res.final_response == ""
    mock_runtime.run.assert_not_called()
    mock_tts.speak.assert_not_called()


@pytest.mark.anyio
async def test_reflection_guard_resets_for_new_voice_session():
    """Verify that reset_session clears in-memory reflection state."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="The answer is 42, sir.", final_response="Ack", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(
        text="The answer is 42, sir.",
        latency_sec=0.01,
        audio_duration_sec=1.0,
        is_empty=False,
    )
    mock_tts = MagicMock()
    mock_tts.speak.return_value = TTSResult(text="Ack", latency_sec=0.01, audio_duration_sec=1.0, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "The answer is 42, sir."

    # Reset session for a new voice session
    pipeline.reset_session()
    assert pipeline._last_spoken_response is None

    dummy_audio = np.ones(16000, dtype=np.float32)
    res = await pipeline.execute_turn_from_audio(dummy_audio)

    # Now that the session was reset, the query is NOT suppressed
    assert res.error is None
    mock_runtime.run.assert_called_once_with("The answer is 42, sir.")


@pytest.mark.anyio
@pytest.mark.parametrize("failure_type", ["exception", "unplayed"])
async def test_tts_failure_does_not_incorrectly_update_previous_spoken_response(failure_type):
    """Verify that failed or unplayed TTS does not register in the reflection guard."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(
        return_value=AgentRunResult(prompt="Hi", final_response="Failed speech attempt", completed=True)
    )
    mock_stt = MagicMock()
    mock_stt.transcribe_audio.return_value = STTResult(text="Hi", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False)

    mock_tts = MagicMock()
    if failure_type == "exception":
        mock_tts.speak.side_effect = RuntimeError("Audio device busy")
    else:
        mock_tts.speak.return_value = TTSResult(text="Failed speech attempt", latency_sec=0.0, audio_duration_sec=0.0, played=False)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.0,
    )
    pipeline._last_spoken_response = "Initial verified response"

    dummy_audio = np.ones(16000, dtype=np.float32)
    await pipeline.execute_turn_from_audio(dummy_audio)

    # Must preserve old response and NOT overwrite with unplayed text
    assert pipeline._last_spoken_response == "Initial verified response"



