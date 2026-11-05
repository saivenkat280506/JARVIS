"""Unit tests for Voice STT module and AudioRecorder."""
from typing import Optional
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from voice.stt import AudioRecorder, SpeechToTextEngine, STTResult


def test_audio_recorder_device_discovery():
    with patch("sounddevice.query_devices") as mock_query:
        mock_query.return_value = [
            {"name": "Mic 1", "max_input_channels": 1},
            {"name": "Speaker", "max_input_channels": 0},
        ]
        recorder = AudioRecorder()
        assert recorder.is_input_available() is True
        devices = recorder.get_input_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Mic 1"


def test_audio_recorder_no_devices():
    with patch("sounddevice.query_devices", return_value=[]):
        recorder = AudioRecorder()
        assert recorder.is_input_available() is False
        with pytest.raises(RuntimeError, match="No audio input device available"):
            recorder.record_fixed(1.0)


def test_audio_recorder_record_fixed():
    recorder = AudioRecorder(sample_rate=16000)
    fake_audio = np.zeros((16000, 1), dtype=np.float32)

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.rec", return_value=fake_audio) as mock_rec, \
         patch("sounddevice.wait") as mock_wait:
        result = recorder.record_fixed(1.0)
        mock_rec.assert_called_once_with(16000, samplerate=16000, channels=1, dtype="float32")
        mock_wait.assert_called_once()
        assert result.shape == (16000,)


def test_audio_recorder_push_to_talk():
    import threading

    recorder = AudioRecorder(sample_rate=16000)
    stop_event = threading.Event()
    stop_event.set()  # immediate stop

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream") as mock_stream_cls:
        mock_stream = MagicMock()
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        result = recorder.record_push_to_talk(stop_event, max_duration_sec=1.0)
        assert len(result) == 0


def test_stt_transcribe_empty_or_silence():
    stt = SpeechToTextEngine()

    # None or empty
    res_none = stt.transcribe_audio(np.zeros(0, dtype=np.float32))
    assert res_none.is_empty is True
    assert res_none.text == ""

    # Short audio (< 0.2s)
    short_audio = np.ones(1000, dtype=np.float32) * 0.1  # 1000 samples @ 16kHz = 0.0625s
    res_short = stt.transcribe_audio(short_audio, sample_rate=16000)
    assert res_short.is_empty is True

    # Low amplitude silence (< 0.003 RMS)
    silent_audio = np.zeros(16000, dtype=np.float32)
    res_silent = stt.transcribe_audio(silent_audio, sample_rate=16000)
    assert res_silent.is_empty is True


def test_stt_transcribe_audio_mocked():
    stt = SpeechToTextEngine()

    mock_segment = MagicMock()
    mock_segment.text = "Hello JARVIS"

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_segment], MagicMock())
    stt._model = mock_whisper

    # Create dummy 1.0 second sine wave audio with sufficient amplitude
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    res = stt.transcribe_audio(audio, sample_rate=16000)
    assert res.text == "Hello JARVIS"
    assert res.is_empty is False
    assert res.latency_sec >= 0.0
    assert abs(res.audio_duration_sec - 1.0) < 0.01


def test_stt_transcribe_file_not_found():
    stt = SpeechToTextEngine()
    res = stt.transcribe_file("/nonexistent/audio/file.wav")
    assert res.is_empty is True
    assert "File not found" in (res.error or "")


# ==============================================================================
# Milestone 10B-3: Deterministic Regression Tests for Dynamic Calibration & Endpointing
# ==============================================================================

class MockAudioStream:
    """Deterministic synthetic audio stream for InputStream mocking."""

    def __init__(self, chunks: list[np.ndarray], default_chunk: Optional[np.ndarray] = None):
        self.chunks = list(chunks)
        self.idx = 0
        self.default_chunk = (
            default_chunk
            if default_chunk is not None
            else np.zeros((1600, 1), dtype=np.float32)
        )

    def read(self, chunk_size: int):
        if self.idx < len(self.chunks):
            chunk = self.chunks[self.idx]
            self.idx += 1
            return chunk, False
        return self.default_chunk.copy(), False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def make_chunk(rms: float, size: int = 1600) -> np.ndarray:
    """Generate a constant-amplitude mono audio chunk with precise target RMS."""
    if rms == 0.0:
        return np.zeros((size, 1), dtype=np.float32)
    return np.full((size, 1), rms, dtype=np.float32)


def test_calibration_and_threshold_derivation_math():
    """Verify noise-floor estimation, clamping, and silence hysteresis logic."""
    recorder = AudioRecorder(
        sample_rate=16000,
        noise_floor_multiplier=2.0,
        min_energy_threshold=0.008,
        max_energy_threshold=0.080,
        silence_threshold_multiplier=1.3,
    )

    # 1. Outlier rejection using 25th percentile
    energies_with_spike = [0.003, 0.080, 0.003, 0.004]
    floor_spike = recorder.estimate_noise_floor(energies_with_spike)
    assert abs(floor_spike - 0.003) < 1e-4

    # 2. Quiet room: lower bound clamping
    quiet_floor = 0.002
    speech_thresh_quiet = recorder.derive_speech_threshold(quiet_floor)
    assert speech_thresh_quiet == 0.008  # clamped to min_energy_threshold
    silence_thresh_quiet = recorder.derive_silence_threshold(quiet_floor, speech_thresh_quiet)
    assert silence_thresh_quiet < speech_thresh_quiet
    assert silence_thresh_quiet >= quiet_floor

    # 3. Moderate noise floor: proportional scaling
    mod_floor = 0.025
    speech_thresh_mod = recorder.derive_speech_threshold(mod_floor)
    assert speech_thresh_mod == 0.050
    silence_thresh_mod = recorder.derive_silence_threshold(mod_floor, speech_thresh_mod)
    assert abs(silence_thresh_mod - 0.0325) < 1e-4
    assert silence_thresh_mod < speech_thresh_mod

    # 4. Very high noise floor: upper bound clamping & hysteresis
    high_floor = 0.050
    speech_thresh_high = recorder.derive_speech_threshold(high_floor)
    assert speech_thresh_high == 0.080  # clamped to max_energy_threshold
    silence_thresh_high = recorder.derive_silence_threshold(high_floor, speech_thresh_high)
    assert silence_thresh_high <= speech_thresh_high * 0.85


def test_record_quiet_ambient_followed_by_speech():
    """Test 1: Quiet ambient noise followed by normal speech."""
    recorder = AudioRecorder(sample_rate=16000)
    # 4 calibration chunks (0.002), 5 speech chunks (0.060), 12 silence chunks (0.002)
    chunks = (
        [make_chunk(0.002) for _ in range(4)]
        + [make_chunk(0.060) for _ in range(5)]
        + [make_chunk(0.002) for _ in range(12)]
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks)):
        result = recorder.record_until_silence(silence_duration_sec=1.2, max_duration_sec=5.0)

    assert len(result) > 0
    # Should contain 3 lead chunks + 5 speech chunks + 12 silence chunks = 20 chunks = 32000 samples
    assert len(result) == 20 * 1600
    assert np.max(result) >= 0.059


def test_record_higher_ambient_noise_followed_by_speech():
    """Test 2: A higher ambient noise floor followed by distinguishable speech."""
    recorder = AudioRecorder(sample_rate=16000)
    # 4 calibration chunks (0.025), 2 pre-speech ambient (0.025),
    # 5 speech chunks (0.070), 12 post-speech ambient chunks (0.025)
    chunks = (
        [make_chunk(0.025) for _ in range(4)]
        + [make_chunk(0.025) for _ in range(2)]
        + [make_chunk(0.070) for _ in range(5)]
        + [make_chunk(0.025) for _ in range(12)]
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks, default_chunk=make_chunk(0.025))):
        result = recorder.record_until_silence(silence_duration_sec=1.2, max_duration_sec=5.0)

    assert len(result) > 0
    # Higher ambient noise (0.025) was recognized as silence post-speech (< 0.0325 threshold)
    # So recording concludes at 20 chunks rather than running to max_duration
    assert len(result) == 20 * 1600
    assert np.max(result) >= 0.069


def test_record_single_transient_during_calibration():
    """Test 3: A single loud transient during calibration is rejected."""
    recorder = AudioRecorder(sample_rate=16000)
    # Calibration has 1 spike of 0.090, then quiet noise (0.004) for all subsequent chunks
    chunks = [
        make_chunk(0.004),
        make_chunk(0.090),  # transient pop/spike
        make_chunk(0.004),
        make_chunk(0.004),
    ] + [make_chunk(0.004) for _ in range(16)]

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks, default_chunk=make_chunk(0.004))):
        result = recorder.record_until_silence(max_duration_sec=2.0)

    # Transient does not trigger speech; entire recording completes with no speech
    assert len(result) == 0


def test_record_quiet_speech_above_calibrated_threshold():
    """Test 4: Quiet speech above the calibrated threshold."""
    recorder = AudioRecorder(sample_rate=16000)
    # Quiet room (0.002 floor -> speech threshold 0.008)
    # Speech at 0.012 is above 0.008 and should trigger
    chunks = (
        [make_chunk(0.002) for _ in range(4)]
        + [make_chunk(0.012) for _ in range(4)]
        + [make_chunk(0.002) for _ in range(12)]
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks)):
        result = recorder.record_until_silence(silence_duration_sec=1.2, max_duration_sec=5.0)

    assert len(result) > 0
    assert np.max(result) >= 0.011


def test_record_continuous_ambient_noise_no_speech():
    """Test 5: Continuous ambient noise with no speech returns empty array."""
    recorder = AudioRecorder(sample_rate=16000)
    chunks = [make_chunk(0.020) for _ in range(15)]

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks, default_chunk=make_chunk(0.020))):
        result = recorder.record_until_silence(max_duration_sec=1.5)

    assert len(result) == 0


def test_record_brief_pauses_between_speech_chunks():
    """Test 6: Brief pauses between speech chunks do not cut off utterance."""
    recorder = AudioRecorder(sample_rate=16000)
    # Word 1 (4 chunks), Pause (3 chunks = 0.3s < 1.2s silence limit), Word 2 (4 chunks), Silence (12 chunks)
    chunks = (
        [make_chunk(0.003) for _ in range(4)]  # calib
        + [make_chunk(0.060) for _ in range(4)]  # word 1
        + [make_chunk(0.003) for _ in range(3)]  # pause
        + [make_chunk(0.060) for _ in range(4)]  # word 2
        + [make_chunk(0.003) for _ in range(12)] # end silence
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks)):
        result = recorder.record_until_silence(silence_duration_sec=1.2, max_duration_sec=5.0)

    # All components preserved: 3 lead + 4 word1 + 3 pause + 4 word2 + 12 silence = 26 chunks = 41600 samples
    assert len(result) == 26 * 1600


def test_record_max_duration_enforced():
    """Test 7: Maximum-duration enforcement in continuous noise/speech."""
    recorder = AudioRecorder(sample_rate=16000)

    # 1. Continuous speech with fixed threshold terminates at exactly max_duration
    stream_fixed = MockAudioStream([], default_chunk=make_chunk(0.060))
    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=stream_fixed):
        result_fixed = recorder.record_until_silence(
            max_duration_sec=1.0,
            energy_threshold=0.015,
            dynamic_calibration=False,
        )
    assert len(result_fixed) == 16000

    # 2. Continuous speech with dynamic calibration strictly respects max_duration cap
    calib_chunks = [make_chunk(0.002) for _ in range(4)]
    stream_calib = MockAudioStream(calib_chunks, default_chunk=make_chunk(0.060))
    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=stream_calib):
        result_calib = recorder.record_until_silence(max_duration_sec=1.0)
    assert 0 < len(result_calib) <= 16000


def test_record_pre_speech_audio_preservation():
    """Test 8: Pre-speech audio preservation ensures initial speech onsets are not clipped."""
    recorder = AudioRecorder(sample_rate=16000)
    # Marker chunks to verify exact lead preservation
    lead_marker_1 = np.full((1600, 1), 0.0021, dtype=np.float32)
    lead_marker_2 = np.full((1600, 1), 0.0022, dtype=np.float32)
    lead_marker_3 = np.full((1600, 1), 0.0023, dtype=np.float32)
    speech_chunk = np.full((1600, 1), 0.055, dtype=np.float32)

    chunks = (
        [make_chunk(0.002) for _ in range(4)]  # calib
        + [lead_marker_1, lead_marker_2, lead_marker_3]  # exactly 3 lead chunks
        + [speech_chunk]
        + [make_chunk(0.002) for _ in range(12)]  # silence
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks)):
        result = recorder.record_until_silence(
            silence_duration_sec=1.2,
            max_duration_sec=5.0,
            speech_lead_chunks=3,
        )

    assert len(result) == 16 * 1600
    # First chunk in result must be lead_marker_1
    assert abs(result[0] - 0.0021) < 1e-5
    # Second chunk must be lead_marker_2
    assert abs(result[1600] - 0.0022) < 1e-5
    # Third chunk must be lead_marker_3
    assert abs(result[3200] - 0.0023) < 1e-5
    # Fourth chunk must be the speech chunk
    assert abs(result[4800] - 0.055) < 1e-5


def test_record_empty_recordings_and_capture_errors():
    """Test 9: Empty recordings and capture errors fail gracefully."""
    recorder = AudioRecorder(sample_rate=16000)

    # 1. No input devices
    with patch.object(recorder, "is_input_available", return_value=False):
        with pytest.raises(RuntimeError, match="No audio input device available"):
            recorder.record_until_silence()

    # 2. Hardware stream exception
    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", side_effect=OSError("CoreAudio device disconnected")):
        with pytest.raises(RuntimeError, match="Microphone recording failed"):
            recorder.record_until_silence()


def test_record_fixed_threshold_fallback_compatibility():
    """Test 10: Fixed threshold override when dynamic_calibration=False."""
    recorder = AudioRecorder(sample_rate=16000)

    # With dynamic_calibration=False, energy_threshold=0.030 should be used directly
    chunks = (
        [make_chunk(0.020) for _ in range(3)]  # below 0.030 -> ignored
        + [make_chunk(0.040) for _ in range(3)]  # above 0.030 -> triggers
        + [make_chunk(0.010) for _ in range(12)] # silence
    )

    with patch.object(recorder, "is_input_available", return_value=True), \
         patch("sounddevice.InputStream", return_value=MockAudioStream(chunks)):
        result = recorder.record_until_silence(
            silence_duration_sec=1.2,
            max_duration_sec=5.0,
            energy_threshold=0.030,
            dynamic_calibration=False,
        )

    assert len(result) > 0
    assert np.max(result) >= 0.039
