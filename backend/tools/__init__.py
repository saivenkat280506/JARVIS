"""Tools package for JARVIS local capabilities."""
from tools.browser import (
    BrowserController,
    default_browser_controller,
    normalize_and_validate_url,
    open_website,
)
from tools.music import (
    MusicController,
    MusicState,
    default_music_controller,
    pause_music,
    play_music,
    resume_music,
    stop_music,
)
from tools.reminders import (
    CancelReminderArgs,
    CreateReminderArgs,
    ListRemindersArgs,
    cancel_reminder,
    create_reminder,
    list_reminders,
)

__all__ = [
    "MusicController",
    "MusicState",
    "default_music_controller",
    "play_music",
    "pause_music",
    "resume_music",
    "stop_music",
    "BrowserController",
    "default_browser_controller",
    "normalize_and_validate_url",
    "open_website",
    "create_reminder",
    "list_reminders",
    "cancel_reminder",
    "CreateReminderArgs",
    "ListRemindersArgs",
    "CancelReminderArgs",
]

