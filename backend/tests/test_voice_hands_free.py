"""Comprehensive deterministic unit and integration tests for JARVIS Continuous Hands-Free Voice Loop.
Milestone 10B-5.

Covers:
1. Multiple hands-free turns without pressing Enter.
2. Conversation ID reused across all turns in a session.
3. Reflection suppression followed by normal listening.
4. Cooldown completed before the next recording starts.
5. No-speech turns do not cause a busy loop.
6. Microphone unavailable at startup and device failure mid-session.
7. STT, runtime, and TTS errors followed by safe recovery.
8. Ctrl+C/cancellation releases resources and restores music state.
9. Pending approvals cannot be approved by ambient speech or assistant reflection.
10. Existing push-to-talk behavior remains unchanged.
11. Existing latency metrics emit once per attempted turn.
12. No automatic replay of consequential actions after failure or restart.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent.runtime import AgentRunResult, AgentRuntime, ToolExecutionRecord
from config.settings import settings
from security.approval import ActionApprovalManager, ApprovalStatus
from storage.action_ledger import ActionLedger, ActionLedgerStatus
from storage.conversation import ConversationStorage
from storage.tasks import TaskStorage, TaskStatus
from tools.music import MusicController, MusicState
from voice.pipeline import VoicePipeline, VoiceTurnResult
from voice.stt import AudioRecorder, SpeechToTextEngine, STTResult
from voice.tts import TextToSpeechEngine, TTSResult


# ---------------------------------------------------------------------------
# Test 1: Multiple hands-free turns without pressing Enter
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_multiple_hands_free_turns_without_enter():
    """Verify that multiple turns are executed sequentially without Enter key interaction."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=[
        AgentRunResult(prompt="Turn 1", final_response="Answer 1", messages=[], rounds_used=1, completed=True),
        AgentRunResult(prompt="Turn 2", final_response="Answer 2", messages=[], rounds_used=1, completed=True),
        AgentRunResult(prompt="Turn 3", final_response="Answer 3", messages=[], rounds_used=1, completed=True),
    ])

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    dummy_audio = np.ones(16000, dtype=np.float32)
    mock_recorder.record_until_silence.return_value = dummy_audio

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.side_effect = [
        STTResult(text="Turn 1", latency_sec=0.1, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Turn 2", latency_sec=0.1, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Turn 3", latency_sec=0.1, audio_duration_sec=1.0, is_empty=False),
    ]

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.side_effect = [
        TTSResult(text="Answer 1", latency_sec=0.1, audio_duration_sec=1.0, played=True),
        TTSResult(text="Answer 2", latency_sec=0.1, audio_duration_sec=1.0, played=True),
        TTSResult(text="Answer 3", latency_sec=0.1, audio_duration_sec=1.0, played=True),
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    results = await pipeline.run_hands_free_loop(
        conversation_id="conv_test_multi",
        max_turns=3,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 3
    assert [r.user_query for r in results] == ["Turn 1", "Turn 2", "Turn 3"]
    assert [r.final_response for r in results] == ["Answer 1", "Answer 2", "Answer 3"]
    assert mock_recorder.record_until_silence.call_count == 3
    assert mock_stt.transcribe_audio.call_count == 3
    assert mock_runtime.run.call_count == 3
    assert mock_tts.speak.call_count == 3


# ---------------------------------------------------------------------------
# Test 2: One conversation ID reused across all turns
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_conversation_id_reused_across_all_turns():
    """Verify that a single conversation ID persists across all turns in a hands-free session."""
    session_cid = "conv_hands_free_stable_001"
    recorded_cids: list[str] = []

    async def mock_run(prompt: str, conversation_id: Optional[str] = None):
        recorded_cids.append(conversation_id or "")
        return AgentRunResult(
            prompt=prompt,
            final_response=f"Response to {prompt}",
            conversation_id=conversation_id,
            messages=[],
            rounds_used=1,
            completed=True,
        )

    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=mock_run)

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.side_effect = [
        STTResult(text="Query A", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Query B", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Query C", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
    ]

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(text="Response", latency_sec=0.05, audio_duration_sec=0.5, played=True)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    results = await pipeline.run_hands_free_loop(
        conversation_id=session_cid,
        max_turns=3,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 3
    for r in results:
        assert r.conversation_id == session_cid
    assert recorded_cids == [session_cid, session_cid, session_cid]


# ---------------------------------------------------------------------------
# Test 3: Reflection suppression followed by normal listening
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_reflection_suppression_followed_by_normal_listening():
    """Verify that an echoed assistant response is suppressed without invoking runtime,
    and subsequent distinct speech is processed normally.
    """
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=[
        AgentRunResult(prompt="Turn 1", final_response="The weather is sunny in Paris today.", messages=[], rounds_used=1, completed=True),
        AgentRunResult(prompt="Turn 3", final_response="Setting reminder for 5 PM.", messages=[], rounds_used=1, completed=True),
    ])

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.side_effect = [
        STTResult(text="What is the weather?", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        # Reflection echo of previous assistant speech:
        STTResult(text="The weather is sunny in Paris today.", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        # Legitimate next command:
        STTResult(text="Set a reminder for 5 PM.", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
    ]

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.side_effect = [
        TTSResult(text="The weather is sunny in Paris today.", latency_sec=0.05, audio_duration_sec=1.0, played=True),
        TTSResult(text="Setting reminder for 5 PM.", latency_sec=0.05, audio_duration_sec=1.0, played=True),
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    results = await pipeline.run_hands_free_loop(
        max_turns=3,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 3
    # Turn 1: normal
    assert results[0].success is True
    assert results[0].user_query == "What is the weather?"
    # Turn 2: suppressed reflection
    assert results[1].error == "Transcript suppressed by reflection guard"
    assert results[1].final_response == ""
    # Turn 3: normal follow-up
    assert results[2].success is True
    assert results[2].user_query == "Set a reminder for 5 PM."

    # Runtime must have been invoked ONLY 2 times (never for turn 2!)
    assert mock_runtime.run.call_count == 2
    mock_runtime.run.assert_any_call("What is the weather?", conversation_id=pipeline.conversation_id)
    mock_runtime.run.assert_any_call("Set a reminder for 5 PM.", conversation_id=pipeline.conversation_id)


# ---------------------------------------------------------------------------
# Test 4: Cooldown completed before the next recording starts
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_cooldown_completed_before_next_recording_starts():
    """Verify that post-playback cooldown finishes before recording starts for the next turn."""
    cooldown_sec = 0.08
    playback_ended_at: list[float] = []
    recording_started_at: list[float] = []

    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(return_value=AgentRunResult(
        prompt="Query",
        final_response="Response",
        messages=[],
        rounds_used=1,
        completed=True,
    ))

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000

    def fake_record(*args, **kwargs):
        recording_started_at.append(time.monotonic())
        return np.ones(16000, dtype=np.float32)

    mock_recorder.record_until_silence.side_effect = fake_record

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Query", latency_sec=0.01, audio_duration_sec=1.0, is_empty=False
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)

    def fake_speak(text: str):
        playback_ended_at.append(time.monotonic())
        return TTSResult(text=text, latency_sec=0.01, audio_duration_sec=0.5, played=True)

    mock_tts.speak.side_effect = fake_speak

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=cooldown_sec,
    )

    results = await pipeline.run_hands_free_loop(
        max_turns=2,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 2
    assert len(playback_ended_at) == 2
    assert len(recording_started_at) == 2

    # Recording for Turn 2 must have started AFTER playback for Turn 1 finished + cooldown
    gap = recording_started_at[1] - playback_ended_at[0]
    assert gap >= (cooldown_sec - 0.02)  # allow minor clock jitter


# ---------------------------------------------------------------------------
# Test 5: No-speech turns do not cause a busy loop
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_no_speech_turns_do_not_cause_busy_loop():
    """Verify that empty/no-speech turns apply backoff delay to prevent CPU spinning."""
    backoff_delay = 0.05
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock()

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.zeros(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="", latency_sec=0.01, audio_duration_sec=1.0, is_empty=True
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    t0 = time.monotonic()
    results = await pipeline.run_hands_free_loop(
        max_turns=3,
        backoff_on_no_speech_sec=backoff_delay,
    )
    elapsed = time.monotonic() - t0

    assert len(results) == 3
    for r in results:
        assert r.error == "No speech detected"
    # Backoff sleep was applied for each no-speech turn
    assert elapsed >= (3 * backoff_delay * 0.8)
    assert mock_runtime.run.call_count == 0


# ---------------------------------------------------------------------------
# Test 6: Microphone unavailable at startup and device failure mid-session
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_microphone_unavailable_at_startup_and_mid_session():
    """Verify safe failure reporting when microphone is absent at startup or fails mid-session."""
    mock_runtime = MagicMock()
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_tts = MagicMock(spec=TextToSpeechEngine)

    # 1. Startup failure
    mock_recorder.is_input_available.return_value = False
    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    states: list[str] = []
    startup_res = await pipeline.run_hands_free_loop(
        on_state_change=states.append,
    )
    assert startup_res == []
    assert "device_unavailable" in states

    # 2. Mid-session device failure with bounded retry
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.side_effect = RuntimeError("No audio input device available.")

    pipeline_mid = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    mid_states: list[str] = []
    mid_res = await pipeline_mid.run_hands_free_loop(
        max_consecutive_device_errors=3,
        on_state_change=mid_states.append,
    )

    # Stops after 3 consecutive failures
    assert len(mid_res) == 3
    for r in mid_res:
        assert "Microphone recording error" in (r.error or "")
    assert "device_error" in mid_states


# ---------------------------------------------------------------------------
# Test 7: STT, runtime, and TTS errors followed by safe recovery
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_stt_runtime_and_tts_errors_followed_by_safe_recovery():
    """Verify that pipeline recovers safely from transient STT, runtime, and TTS errors."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(side_effect=[
        # Turn 2: Runtime failure
        RuntimeError("Ollama inference timeout"),
        # Turn 3: Normal runtime
        AgentRunResult(prompt="Turn 3", final_response="Answer 3", messages=[], rounds_used=1, completed=True),
        # Turn 4: Normal runtime
        AgentRunResult(prompt="Turn 4", final_response="Answer 4", messages=[], rounds_used=1, completed=True),
    ])

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.side_effect = [
        # Turn 1: STT failure
        RuntimeError("Whisper decode error"),
        # Turn 2: Successful STT
        STTResult(text="Turn 2", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        # Turn 3: Successful STT
        STTResult(text="Turn 3", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        # Turn 4: Successful STT
        STTResult(text="Turn 4", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
    ]

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.side_effect = [
        # Turn 3: TTS failure
        RuntimeError("TTS audio playback broken"),
        # Turn 4: TTS success
        TTSResult(text="Answer 4", latency_sec=0.05, audio_duration_sec=1.0, played=True),
    ]

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    results = await pipeline.run_hands_free_loop(
        max_turns=4,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 4
    # Turn 1: STT error handled cleanly
    assert results[0].success is False
    assert "Speech recognition error" in (results[0].error or "")
    # Turn 2: Runtime error handled cleanly
    assert results[1].success is False
    assert "Agent inference failure" in (results[1].error or "")
    # Turn 3: TTS error handled cleanly without crashing turn
    assert results[2].user_query == "Turn 3"
    assert results[2].final_response == "Answer 3"
    # Turn 4: Completely succeeded
    assert results[3].success is True
    assert results[3].user_query == "Turn 4"
    assert results[3].final_response == "Answer 4"


# ---------------------------------------------------------------------------
# Test 8: Ctrl+C / cancellation releases resources and restores music state
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_cancellation_releases_resources_and_restores_music():
    """Verify that cancellation stops loop, releases pipeline lock, and restores music state."""
    state = {"val": MusicState.PLAYING}
    mock_music = MagicMock(spec=MusicController)
    mock_music.get_state.side_effect = lambda: state["val"]
    def fake_pause():
        state["val"] = MusicState.PAUSED
        return {"success": True}
    def fake_resume():
        state["val"] = MusicState.PLAYING
        return {"success": True}
    mock_music.pause.side_effect = fake_pause
    mock_music.resume.side_effect = fake_resume

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000

    def slow_record(*args, **kwargs):
        raise asyncio.CancelledError()

    mock_recorder.record_until_silence.side_effect = slow_record

    mock_runtime = MagicMock()
    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_tts = MagicMock(spec=TextToSpeechEngine)

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        music_controller=mock_music,
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_hands_free_loop()

    # Audio lifecycle lock must be released
    assert pipeline._lock.locked() is False
    # TTS playback must be explicitly stopped
    mock_tts.stop_playback.assert_called()
    # Music must have been resumed
    mock_music.resume.assert_called_once()


# ---------------------------------------------------------------------------
# Test 9: Pending approvals cannot be approved by ambient speech or assistant reflection
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_pending_approvals_cannot_be_approved_by_ambient_speech_or_assistant_reflection(tmp_path: Path):
    """Verify that ambient speech or assistant self-capture cannot authorize pending actions."""
    db_path = tmp_path / "voice_approval_test.db"
    approval_mgr = ActionApprovalManager(db_path=db_path)
    task_storage = TaskStorage(db_path=db_path)
    conv_storage = ConversationStorage(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path)

    # Register a pending consequential action
    pending = approval_mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="submit_payment",
        target="#pay-btn",
        arguments={"amount": 100, "currency": "USD"},
        description="Submit payment of $100",
    )
    # Register task waiting for approval
    task = task_storage.create_task(
        task_type="payment",
        description="Process user payment",
        conversation_id="conv_voice_approval",
        status=TaskStatus.WAITING_FOR_APPROVAL,
        metadata={"pending_action_id": pending.action_id},
    )

    mock_client = MagicMock()
    runtime = AgentRuntime(
        client=mock_client,
        storage=conv_storage,
        approval_manager=approval_mgr,
        task_storage=task_storage,
        action_ledger=action_ledger,
    )

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    # Sequence of utterances:
    # 1. Assistant echo: "Action requires your approval: Submit payment of $100"
    # 2. Ambient speech: "Nice weather we're having today"
    # 3. Explicit user confirmation: "Confirm"
    mock_stt.transcribe_audio.side_effect = [
        STTResult(text="Action requires your approval: Submit payment of $100", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Nice weather we're having today", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
        STTResult(text="Confirm", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False),
    ]

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(text="Response", latency_sec=0.05, audio_duration_sec=0.5, played=True)

    pipeline = VoicePipeline(
        runtime=runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
        conversation_id="conv_voice_approval",
    )
    # Set previous assistant response to test reflection guard
    pipeline._last_spoken_response = "Action requires your approval: Submit payment of $100"

    # Turn 1: Echoed assistant speech
    res1 = await pipeline.execute_voice_turn(conversation_id="conv_voice_approval")
    assert res1.error == "Transcript suppressed by reflection guard"
    # Assert action remains PENDING and task remains WAITING_FOR_APPROVAL
    pa_after_1 = approval_mgr.get_pending_action(pending.action_id)
    assert pa_after_1 is not None and pa_after_1.status == ApprovalStatus.PENDING
    t_after_1 = task_storage.get_task(task.id)
    assert t_after_1 is not None and t_after_1.status == TaskStatus.WAITING_FOR_APPROVAL

    # Turn 2: Ambient unrelated speech
    res2 = await pipeline.execute_voice_turn(conversation_id="conv_voice_approval")
    assert res2.success is False
    assert "awaiting your explicit authorization" in res2.final_response.lower()
    # Action remains PENDING, task remains WAITING_FOR_APPROVAL
    pa_after_2 = approval_mgr.get_pending_action(pending.action_id)
    assert pa_after_2 is not None and pa_after_2.status == ApprovalStatus.PENDING
    t_after_2 = task_storage.get_task(task.id)
    assert t_after_2 is not None and t_after_2.status == TaskStatus.WAITING_FOR_APPROVAL

    # Turn 3: Explicit trusted confirmation ("Confirm")
    res3 = await pipeline.execute_voice_turn(conversation_id="conv_voice_approval")
    # Action must now be APPROVED
    pa_after_3 = approval_mgr.get_pending_action(pending.action_id)
    assert pa_after_3 is not None and pa_after_3.status == ApprovalStatus.APPROVED


# ---------------------------------------------------------------------------
# Test 10: Existing push-to-talk behavior remains unchanged
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_push_to_talk_mode_remains_unchanged():
    """Verify that push-to-talk mode executes correctly and independently of hands-free loop."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(return_value=AgentRunResult(
        prompt="PTT query",
        final_response="PTT response",
        messages=[],
        rounds_used=1,
        completed=True,
    ))

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.sample_rate = 16000
    mock_recorder.record_push_to_talk.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="PTT query", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(
        text="PTT response", latency_sec=0.05, audio_duration_sec=0.5, played=True
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
    )

    stop_evt = threading.Event()
    stop_evt.set()

    turn_res = await pipeline.execute_push_to_talk_turn(
        stop_event=stop_evt,
        conversation_id="conv_ptt_test",
    )

    assert turn_res.success is True
    assert turn_res.user_query == "PTT query"
    assert turn_res.final_response == "PTT response"
    assert turn_res.metrics is not None
    assert turn_res.metrics.turn_type == "push_to_talk"
    mock_recorder.record_push_to_talk.assert_called_once()


# ---------------------------------------------------------------------------
# Test 11: Existing latency metrics emit once per attempted turn
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_latency_metrics_emitted_once_per_attempted_turn():
    """Verify that structured latency metrics are populated and logged for each turn."""
    mock_runtime = MagicMock()
    mock_runtime.run = AsyncMock(return_value=AgentRunResult(
        prompt="Test metrics",
        final_response="Metrics response",
        messages=[],
        rounds_used=1,
        completed=True,
    ))

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Test metrics", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(
        text="Metrics response", latency_sec=0.05, audio_duration_sec=0.5, played=True
    )

    pipeline = VoicePipeline(
        runtime=mock_runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    results = await pipeline.run_hands_free_loop(
        max_turns=2,
        backoff_on_no_speech_sec=0.01,
    )

    assert len(results) == 2
    for r in results:
        assert r.metrics is not None
        assert r.metrics.turn_type == "voice"
        assert r.metrics.audio_capture_ms is not None
        assert r.metrics.stt_transcription_ms is not None
        assert r.metrics.agent_runtime_ms is not None
        assert r.metrics.total_turn_ms > 0
        log_line = r.metrics.to_compact_log()
        # Verify no sensitive data leakage
        assert "Test metrics" not in log_line
        assert "Metrics response" not in log_line


# ---------------------------------------------------------------------------
# Test 12: No automatic replay of consequential actions after failure
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_no_automatic_replay_of_consequential_actions(tmp_path: Path):
    """Verify that consequential actions with uncertain or failed outcome are never automatically replayed."""
    db_path = tmp_path / "ledger_no_replay.db"
    approval_mgr = ActionApprovalManager(db_path=db_path)
    task_storage = TaskStorage(db_path=db_path)
    conv_storage = ConversationStorage(db_path=db_path)
    ledger = ActionLedger(db_path=db_path)

    # Create an action attempt in OUTCOME_UNKNOWN
    attempt, _ = ledger.prepare_action(
        task_id="task_uncertain_1",
        step_name="browser_submit_pay_123456",
        tool_name="browser_interact",
        arguments={"action": "click", "selector": "#pay-btn"},
        action_id="act_uncertain_999",
        initial_status=ActionLedgerStatus.EXECUTING,
    )
    ledger.update_status(attempt.id, ActionLedgerStatus.OUTCOME_UNKNOWN, status_message="Network dropped")

    mock_client = MagicMock()
    runtime = AgentRuntime(
        client=mock_client,
        storage=conv_storage,
        approval_manager=approval_mgr,
        task_storage=task_storage,
        action_ledger=ledger,
    )

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    mock_recorder.record_until_silence.return_value = np.ones(16000, dtype=np.float32)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Proceed anyway", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(text="Blocked", latency_sec=0.05, audio_duration_sec=0.5, played=True)

    pipeline = VoicePipeline(
        runtime=runtime,
        recorder=mock_recorder,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        post_playback_cooldown_sec=0.01,
    )

    # Attempt to execute turn with prompt "Proceed anyway"
    turn_res = await pipeline.execute_voice_turn(conversation_id="conv_uncertain")

    # Runtime must block automatic replay
    latest_att = ledger.get_latest_attempt("task_uncertain_1:browser_submit_pay_123456")
    assert latest_att is not None
    # No new execution attempt created or succeeded
    assert latest_att.status == ActionLedgerStatus.OUTCOME_UNKNOWN
