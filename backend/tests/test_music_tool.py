"""Unit tests for the macOS MusicController without playing real audio."""
from pathlib import Path
import signal
from typing import Any, List
import pytest

from tools.music import MusicController, MusicState


class MockProcess:
    """Mock subprocess.Popen object for testing MusicController lifecycle."""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self._returncode: int | None = None
        self.signals_received: List[int] = []
        self.killed = False

    def poll(self) -> int | None:
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def terminate(self) -> None:
        self._returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode


def test_music_play_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_track = tmp_path / "test_track.mp3"
    fake_track.write_bytes(b"FAKE_AUDIO_DATA")

    mock_proc = MockProcess()

    def fake_launcher(args: list[str], **kwargs: Any) -> MockProcess:
        return mock_proc

    # Monkeypatch os.kill to capture signals without sending real POSIX signals
    captured_signals: list[tuple[int, int]] = []

    def fake_os_kill(pid: int, sig: int) -> None:
        captured_signals.append((pid, sig))

    monkeypatch.setattr("os.kill", fake_os_kill)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    controller = MusicController(
        music_path=fake_track,
        player_binary="/usr/bin/afplay",
        process_launcher=fake_launcher,
    )

    assert controller.get_state() == MusicState.STOPPED

    # 1. Play
    res_play = controller.play()
    assert res_play["success"] is True
    assert res_play["status"] == "playing"
    assert controller.get_state() == MusicState.PLAYING

    # 2. Repeated Play
    res_repeat = controller.play()
    assert res_repeat["success"] is True
    assert res_repeat["status"] == "already_playing"

    # 3. Pause
    res_pause = controller.pause()
    assert res_pause["success"] is True
    assert res_pause["status"] == "paused"
    assert controller.get_state() == MusicState.PAUSED
    assert (mock_proc.pid, signal.SIGSTOP) in captured_signals

    # 4. Repeated Pause
    res_pause_repeat = controller.pause()
    assert res_pause_repeat["success"] is True
    assert res_pause_repeat["status"] == "already_paused"

    # 5. Resume
    res_resume = controller.resume()
    assert res_resume["success"] is True
    assert res_resume["status"] == "resumed"
    assert controller.get_state() == MusicState.PLAYING
    assert (mock_proc.pid, signal.SIGCONT) in captured_signals

    # 6. Stop
    res_stop = controller.stop()
    assert res_stop["success"] is True
    assert res_stop["status"] == "stopped"
    assert controller.get_state() == MusicState.STOPPED
    assert mock_proc.killed is True

    # 7. Repeated Stop
    res_stop_repeat = controller.stop()
    assert res_stop_repeat["success"] is True
    assert res_stop_repeat["status"] == "already_stopped"


def test_music_invalid_transitions(tmp_path: Path):
    fake_track = tmp_path / "test_track.mp3"
    fake_track.write_bytes(b"FAKE_AUDIO_DATA")

    controller = MusicController(
        music_path=fake_track,
        player_binary="/usr/bin/afplay",
    )

    # Pause while stopped
    res_pause = controller.pause()
    assert res_pause["success"] is False
    assert "Cannot pause: no music is currently playing" in res_pause["error"]

    # Resume while stopped
    res_resume = controller.resume()
    assert res_resume["success"] is False
    assert "Cannot resume: music is stopped" in res_resume["error"]


def test_music_missing_asset(tmp_path: Path):
    missing_track = tmp_path / "does_not_exist.mp3"
    controller = MusicController(
        music_path=missing_track,
        player_binary="/usr/bin/afplay",
    )

    res = controller.play()
    assert res["success"] is False
    assert "not found" in res["error"]
    assert controller.get_state() == MusicState.STOPPED


def test_music_missing_player_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_track = tmp_path / "track.mp3"
    fake_track.write_bytes(b"DATA")

    monkeypatch.setattr("os.path.exists", lambda path: False)

    controller = MusicController(
        music_path=fake_track,
        player_binary="/nonexistent/afplay",
    )

    res = controller.play()
    assert res["success"] is False
    assert "player binary '/nonexistent/afplay' not found" in res["error"]


def test_music_launcher_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_track = tmp_path / "track.mp3"
    fake_track.write_bytes(b"DATA")
    monkeypatch.setattr("os.path.exists", lambda path: True)

    def failing_launcher(*args: Any, **kwargs: Any) -> None:
        raise OSError("Permission denied")

    controller = MusicController(
        music_path=fake_track,
        player_binary="/usr/bin/afplay",
        process_launcher=failing_launcher,
    )

    res = controller.play()
    assert res["success"] is False
    assert "Failed to launch" in res["error"]
    assert controller.get_state() == MusicState.STOPPED


def test_music_natural_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_track = tmp_path / "track.mp3"
    fake_track.write_bytes(b"DATA")
    monkeypatch.setattr("os.path.exists", lambda path: True)

    mock_proc = MockProcess()
    controller = MusicController(
        music_path=fake_track,
        process_launcher=lambda *a, **k: mock_proc,
    )

    controller.play()
    assert controller.get_state() == MusicState.PLAYING

    # Process finishes playing naturally
    mock_proc._returncode = 0
    assert controller.get_state() == MusicState.STOPPED
