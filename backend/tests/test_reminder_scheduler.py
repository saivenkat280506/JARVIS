"""Unit tests for ReminderScheduler background polling and delivery policy."""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import pytest

from scheduler.reminder_scheduler import ReminderScheduler
from storage.reminders import ReminderRecord, ReminderStorage


class MockChannel:
    """Mock notification channel capturing delivery attempts."""

    def __init__(self, should_succeed: bool = True) -> None:
        self.should_succeed = should_succeed
        self.delivered_records: List[ReminderRecord] = []

    def deliver(self, reminder: ReminderRecord) -> Tuple[bool, Optional[str]]:
        if not self.should_succeed:
            return False, "Simulated network/system notification failure"
        self.delivered_records.append(reminder)
        return True, None


@pytest.mark.anyio
async def test_scheduler_duplicate_delivery_prevention(tmp_path: Path):
    db_file = tmp_path / "dedup.db"
    storage = ReminderStorage(db_path=db_file)
    channel = MockChannel(should_succeed=True)
    scheduler = ReminderScheduler(storage=storage, channel=channel)

    now = datetime.now().astimezone()
    # Create reminder due now
    r1 = storage.create_reminder("Standup meeting", now - timedelta(seconds=1), "IST")

    # First poll: delivers the reminder
    results1 = await scheduler.poll_once(as_of=now)
    assert len(results1) == 1
    assert results1[0]["id"] == r1.id
    assert results1[0]["status"] == "delivered"
    assert len(channel.delivered_records) == 1

    # Second poll: does NOT re-deliver
    results2 = await scheduler.poll_once(as_of=now + timedelta(seconds=5))
    assert len(results2) == 0
    assert len(channel.delivered_records) == 1


@pytest.mark.anyio
async def test_scheduler_missed_reminder_restart_policy(tmp_path: Path):
    db_file = tmp_path / "restart.db"
    storage = ReminderStorage(db_path=db_file)
    channel = MockChannel(should_succeed=True)
    scheduler = ReminderScheduler(
        storage=storage, channel=channel, missed_threshold_sec=30.0
    )

    now = datetime.now().astimezone()
    # Reminder scheduled 5 minutes ago (e.g. app was offline during due time)
    r_missed = storage.create_reminder("Scheduled while offline", now - timedelta(minutes=5), "IST")

    results = await scheduler.poll_once(as_of=now)
    assert len(results) == 1
    assert results[0]["was_missed"] is True
    assert "[MISSED REMINDER" in channel.delivered_records[0].text


@pytest.mark.anyio
async def test_scheduler_delivery_failure_handling(tmp_path: Path):
    db_file = tmp_path / "fail.db"
    storage = ReminderStorage(db_path=db_file)
    failing_channel = MockChannel(should_succeed=False)
    scheduler = ReminderScheduler(storage=storage, channel=failing_channel)

    now = datetime.now().astimezone()
    r = storage.create_reminder("Unreachable alert", now - timedelta(seconds=1), "IST")

    results = await scheduler.poll_once(as_of=now)
    assert len(results) == 1
    assert results[0]["status"] == "delivery_failed"
    assert "Simulated network/system" in results[0]["error"]

    # Verify database state recorded the error and kept it pending for retry
    record = storage.get_reminder(r.id)
    assert record is not None
    assert record.status == "pending"
    assert "Simulated" in (record.delivery_error or "")


@pytest.mark.anyio
async def test_scheduler_lifecycle(tmp_path: Path):
    db_file = tmp_path / "lifecycle.db"
    storage = ReminderStorage(db_path=db_file)
    channel = MockChannel(should_succeed=True)
    scheduler = ReminderScheduler(storage=storage, channel=channel, poll_interval_sec=0.05)

    assert scheduler.is_running is False
    scheduler.start()
    assert scheduler.is_running is True

    await asyncio.sleep(0.1)
    await scheduler.stop()
    assert scheduler.is_running is False
