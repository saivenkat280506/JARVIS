"""Unit tests for Voice TTS module, sanitization, and playback."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import numpy as np
import pytest
import torch

from voice.tts import TextToSpeechEngine, TTSResult, sanitize_text_for_speech


def test_sanitize_text_for_speech():
    raw = """
    <think>
    Thinking through the plan: the user wants to calculate something.
    </think>
    ### Calculation Result
    Here is the **calculated** answer: `42`.
    * Step 1: Add 40
    * Step 2: Add 2
    ```python
    x = 40 + 2
    ```
    Visit [Apple](https://apple.com) for more.
    """
    clean = sanitize_text_for_speech(raw)
    assert "<think>" not in clean
    assert "###" not in clean
    assert "**" not in clean
    assert "`" not in clean
    assert "https://apple.com" not in clean
    assert "Here is the calculated answer: 42." in clean
    assert "Apple for more." in clean


def test_sanitize_technical_tokens_for_speech():
    raw_reminder = "Reminder set! You'll be notified in 15 minutes (ID: rem_68558e82)."
    clean_reminder = sanitize_text_for_speech(raw_reminder)
    assert "rem_68558e82" not in clean_reminder
    assert "ID:" not in clean_reminder
    assert "Reminder set! You'll be notified in 15 minutes." == clean_reminder

    raw_time = "It's currently 1:53:07 PM IST (UTC+05:30)."
    clean_time = sanitize_text_for_speech(raw_time)
    assert "UTC" not in clean_time
    assert "It's currently 1:53:07 PM IST." == clean_time

    raw_music = "Now playing 'garage_music.mp3'."
    clean_music = sanitize_text_for_speech(raw_music)
    assert "garage music.mp3" in clean_music
    assert "_" not in clean_music



def test_tts_empty_text():
    tts = TextToSpeechEngine()
    res = tts.speak("   ")
    assert res.played is False
    assert res.audio_duration_sec == 0.0
    assert res.error is None


def test_tts_synthesize_and_speak_mocked(tmp_path):
    tts = TextToSpeechEngine()

    mock_model = MagicMock()
    mock_model.sample_rate = 24000
    # Generate 1 second of dummy float tensor
    dummy_tensor = torch.zeros(24000, dtype=torch.float32)
    mock_model.generate_audio.return_value = dummy_tensor
    tts._model = mock_model
    tts._voice_state = {"state": "ready"}

    created_temp_files = []

    def mock_play(path):
        assert os.path.exists(path)
        created_temp_files.append(path)
        return True

    with patch.object(tts, "_play_audio_file", side_effect=mock_play):
        res = tts.speak("Good day, sir.")
        assert res.played is True
        assert res.latency_sec >= 0.0
        assert abs(res.audio_duration_sec - 1.0) < 0.01
        assert len(created_temp_files) == 1
        # Crucial check: verify temporary file was cleaned up
        assert not os.path.exists(created_temp_files[0])


def test_tts_play_audio_file_afplay():
    tts = TextToSpeechEngine()

    with patch("os.path.isfile", return_value=True), \
         patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        played = tts._play_audio_file("/dummy/path.wav")
        assert played is True
        mock_proc.wait.assert_called_once()


def test_tts_stop_playback():
    tts = TextToSpeechEngine()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # process is running
    tts._current_process = mock_proc

    tts.stop_playback()
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=0.5)
    assert tts._current_process is None
