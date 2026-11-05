"""Lightweight asynchronous scheduler for polling and delivering due reminders."""
from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from scheduler.notifications import (
    DefaultNotificationChannel,
    NotificationChannel,
)
from storage.reminders import (
    ReminderRecord,
    ReminderStatus,
    ReminderStorage,
    default_reminder_storage,
)

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Asynchronous background scheduler delivering due reminders.

    Policy on missed reminders:
    If application was offline and restarts after a reminder's due time, the scheduler
    detects the overdue pending reminder on its first poll, delivers it with a
    prominent '[MISSED / LATE]' prefix, and records it as delivered.
    """

    def __init__(
        self,
        storage: Optional[ReminderStorage] = None,
        channel: Optional[NotificationChannel] = None,
        poll_interval_sec: float = 2.0,
        missed_threshold_sec: float = 30.0,
    ) -> None:
        self.storage = storage or default_reminder_storage
        self.channel = channel or DefaultNotificationChannel()
        self.poll_interval_sec = poll_interval_sec
        self.missed_threshold_sec = missed_threshold_sec

        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._running

    async def poll_once(self, as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Perform a single check for due reminders and deliver them."""
        check_time = as_of or datetime.now().astimezone()
        check_epoch = check_time.timestamp()

        due_reminders = self.storage.get_due_reminders(check_epoch)
        results: List[Dict[str, Any]] = []

        for reminder in due_reminders:
            lateness_sec = check_epoch - reminder.due_timestamp
            is_missed = lateness_sec > self.missed_threshold_sec

            item_to_deliver = reminder
            if is_missed:
                # Mark reminder text with clear missed banner so user knows it was delayed
                item_to_deliver = ReminderRecord(
                    id=reminder.id,
                    text=f"[MISSED REMINDER - Due was: {reminder.due_at}] {reminder.text}",
                    due_at=reminder.due_at,
                    due_timestamp=reminder.due_timestamp,
                    timezone=reminder.timezone,
                    status=reminder.status,
                    created_at=reminder.created_at,
                )

            # Attempt delivery via channel abstraction
            try:
                success, error = self.channel.deliver(item_to_deliver)
            except Exception as exc:
                success = False
                error = str(exc)

            if success:
                self.storage.mark_delivered(
                    reminder.id,
                    delivered_at=check_time,
                    error=None,
                    status=ReminderStatus.DELIVERED.value,
                )
                results.append({
                    "id": reminder.id,
                    "text": reminder.text,
                    "status": "delivered",
                    "was_missed": is_missed,
                    "lateness_sec": round(lateness_sec, 2),
                })
            else:
                self.storage.mark_delivered(
                    reminder.id,
                    delivered_at=check_time,
                    error=error,
                    status=ReminderStatus.PENDING.value,  # Keep pending to retry later
                )
                results.append({
                    "id": reminder.id,
                    "text": reminder.text,
                    "status": "delivery_failed",
                    "error": error,
                })

        return results

    async def _run_loop(self) -> None:
        """Internal background polling loop."""
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                logger.error("Error in reminder scheduler polling loop: %s", exc)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval_sec
                )
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background polling loop cleanly."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None


# Global shared instance
default_scheduler = ReminderScheduler()
