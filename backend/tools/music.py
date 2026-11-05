"""Music playback controller for macOS using native process management."""
from __future__ import annotations

from enum import Enum
import logging
import os
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any, Callable, Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class MusicState(str, Enum):
    """Lifecycle states of the music controller."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class MusicController:
    """Safe, process-backed music player controller designed for macOS.

    Uses macOS native `/usr/bin/afplay` for low-overhead audio playback.
    Process signals (SIGSTOP / SIGCONT) provide real hardware-level pause and resume,
    and process PID polling tracks completion without thread leaks.
    """

    def __init__(
        self,
        music_path: Optional[Path] = None,
        player_binary: str = "/usr/bin/afplay",
        process_launcher: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.music_path = (music_path or settings.music_path).resolve()
        self.player_binary = player_binary
        self._launcher = process_launcher or subprocess.Popen
        self._process: Optional[Any] = None
        self._state: MusicState = MusicState.STOPPED
        self._lock = threading.Lock()

    def get_state(self) -> MusicState:
        """Inspect active process and return current playback state."""
        with self._lock:
            if self._process is not None:
                # If process has finished naturally on its own
                if self._process.poll() is not None:
                    self._process = None
                    self._state = MusicState.STOPPED
            return self._state

    def play(self) -> Dict[str, Any]:
        """Start or restart audio playback."""
        with self._lock:
            # Verify asset file exists
            if not self.music_path.is_file():
                return {
                    "success": False,
                    "error": f"Music asset not found at '{self.music_path}'.",
                    "state": MusicState.STOPPED.value,
                }

            # Check if player binary is available on the system
            if not os.path.exists(self.player_binary):
                return {
                    "success": False,
                    "error": f"Audio player binary '{self.player_binary}' not found.",
                    "state": MusicState.STOPPED.value,
                }

            # Check if process is already running
            if self._process is not None and self._process.poll() is None:
                if self._state == MusicState.PLAYING:
                    return {
                        "success": True,
                        "status": "already_playing",
                        "message": f"Music '{self.music_path.name}' is already playing.",
                        "track": self.music_path.name,
                        "state": self._state.value,
                    }
                elif self._state == MusicState.PAUSED:
                    # If paused, resume playback
                    try:
                        os.kill(self._process.pid, signal.SIGCONT)
                        self._state = MusicState.PLAYING
                        return {
                            "success": True,
                            "status": "resumed",
                            "message": f"Resumed playback of '{self.music_path.name}'.",
                            "track": self.music_path.name,
                            "state": self._state.value,
                        }
                    except OSError as exc:
                        self._process = None
                        self._state = MusicState.STOPPED
                        return {
                            "success": False,
                            "error": f"Failed to resume existing player: {exc}",
                            "state": MusicState.STOPPED.value,
                        }

            # Spawn a new playback process
            try:
                self._process = self._launcher(
                    [self.player_binary, str(self.music_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._state = MusicState.PLAYING
                return {
                    "success": True,
                    "status": "playing",
                    "message": f"Now playing '{self.music_path.name}'.",
                    "track": self.music_path.name,
                    "state": self._state.value,
                }
            except Exception as exc:
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": False,
                    "error": f"Failed to launch audio playback: {exc}",
                    "state": MusicState.STOPPED.value,
                }

    def pause(self) -> Dict[str, Any]:
        """Pause playback using SIGSTOP signal."""
        with self._lock:
            # Re-check poll status
            if self._process is None or self._process.poll() is not None:
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": False,
                    "error": "Cannot pause: no music is currently playing.",
                    "state": MusicState.STOPPED.value,
                }

            if self._state == MusicState.PAUSED:
                return {
                    "success": True,
                    "status": "already_paused",
                    "message": f"Playback of '{self.music_path.name}' is already paused.",
                    "state": MusicState.PAUSED.value,
                }

            try:
                os.kill(self._process.pid, signal.SIGSTOP)
                self._state = MusicState.PAUSED
                return {
                    "success": True,
                    "status": "paused",
                    "message": f"Paused playback of '{self.music_path.name}'.",
                    "track": self.music_path.name,
                    "state": self._state.value,
                }
            except OSError as exc:
                return {
                    "success": False,
                    "error": f"Failed to pause playback process: {exc}",
                    "state": self._state.value,
                }

    def resume(self) -> Dict[str, Any]:
        """Resume playback using SIGCONT signal."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": False,
                    "error": "Cannot resume: music is stopped. Use play_music to start playback.",
                    "state": MusicState.STOPPED.value,
                }

            if self._state == MusicState.PLAYING:
                return {
                    "success": True,
                    "status": "already_playing",
                    "message": f"Playback of '{self.music_path.name}' is already active.",
                    "track": self.music_path.name,
                    "state": MusicState.PLAYING.value,
                }

            try:
                os.kill(self._process.pid, signal.SIGCONT)
                self._state = MusicState.PLAYING
                return {
                    "success": True,
                    "status": "resumed",
                    "message": f"Resumed playback of '{self.music_path.name}'.",
                    "track": self.music_path.name,
                    "state": self._state.value,
                }
            except OSError as exc:
                return {
                    "success": False,
                    "error": f"Failed to resume playback process: {exc}",
                    "state": self._state.value,
                }

    def stop(self) -> Dict[str, Any]:
        """Terminate playback process cleanly."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": True,
                    "status": "already_stopped",
                    "message": "Music playback is already stopped.",
                    "state": MusicState.STOPPED.value,
                }

            try:
                # If paused, SIGKILL terminates immediately without waiting for SIGCONT
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                except Exception:
                    pass
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": True,
                    "status": "stopped",
                    "message": f"Stopped playback of '{self.music_path.name}'.",
                    "state": MusicState.STOPPED.value,
                }
            except Exception as exc:
                self._process = None
                self._state = MusicState.STOPPED
                return {
                    "success": False,
                    "error": f"Error stopping playback process: {exc}",
                    "state": MusicState.STOPPED.value,
                }


# Global shared instance for the runtime
default_music_controller = MusicController()


def play_music() -> Dict[str, Any]:
    """Tool handler for play_music."""
    return default_music_controller.play()


def pause_music() -> Dict[str, Any]:
    """Tool handler for pause_music."""
    return default_music_controller.pause()


def resume_music() -> Dict[str, Any]:
    """Tool handler for resume_music."""
    return default_music_controller.resume()


def stop_music() -> Dict[str, Any]:
    """Tool handler for stop_music."""
    return default_music_controller.stop()
