"""JARVIS Desktop Subsystem.

Provides process lifecycle management, desktop shell integration,
and startup/shutdown supervision for the native macOS desktop application.
"""
from __future__ import annotations

from .lifecycle import DesktopLifecycleManager, InstanceLockError, StartupError

__all__ = ["DesktopLifecycleManager", "InstanceLockError", "StartupError"]
