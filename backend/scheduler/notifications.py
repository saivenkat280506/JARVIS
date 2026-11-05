"""Notification channels for delivering scheduled JARVIS reminders."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional, Protocol, Tuple

from storage.reminders import ReminderRecord

logger = logging.getLogger(__name__)


class NotificationChannel(Protocol):
    """Protocol for delivering reminder alerts."""

    def deliver(self, reminder: ReminderRecord) -> Tuple[bool, Optional[str]]:
        """Deliver the notification. Returns (success_flag, error_message)."""
        ...


class ConsoleNotificationChannel:
    """Logs and prints prominent reminder notifications to the console."""

    def deliver(self, reminder: ReminderRecord) -> Tuple[bool, Optional[str]]:
        border = "=" * 64
        print(f"\n{border}")
        print("  🔔 J.A.R.V.I.S. REMINDER NOTIFICATION")
        print(border)
        print(f"  • ID:       {reminder.id}")
        print(f"  • Due:      {reminder.due_at} ({reminder.timezone})")
        print(f"  • Message:  {reminder.text}")
        print(f"{border}\n")
        return True, None


class MacOSNotificationChannel:
    """Dispatches native desktop banner alerts on macOS via osascript."""

    def __init__(self, osascript_path: str = "/usr/bin/osascript") -> None:
        self.osascript_path = osascript_path

    def deliver(self, reminder: ReminderRecord) -> Tuple[bool, Optional[str]]:
        if not os.path.exists(self.osascript_path):
            return False, f"osascript binary '{self.osascript_path}' not found."

        # Escape backslashes and double quotes for AppleScript literal
        safe_text = reminder.text.replace("\\", "\\\\").replace('"', '\\"')
        safe_title = "J.A.R.V.I.S. Reminder"

        script = (
            f'display notification "{safe_text}" with title "{safe_title}"'
        )

        try:
            res = subprocess.run(
                [self.osascript_path, "-e", script],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0:
                return True, None
            return False, f"osascript exited with code {res.returncode}: {res.stderr.strip()}"
        except Exception as exc:
            return False, f"Failed to execute osascript: {exc}"


class DefaultNotificationChannel:
    """Composite channel delivering to both console and native macOS desktop."""

    def __init__(self) -> None:
        self.console = ConsoleNotificationChannel()
        self.macos = MacOSNotificationChannel()

    def deliver(self, reminder: ReminderRecord) -> Tuple[bool, Optional[str]]:
        # Always output to console
        self.console.deliver(reminder)

        # Attempt macOS notification if available
        macos_success, macos_err = self.macos.deliver(reminder)
        if not macos_success:
            logger.debug("macOS notification dispatch notice: %s", macos_err)

        # As long as console delivered, notification delivery succeeded
        return True, None
