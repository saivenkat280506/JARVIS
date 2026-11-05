"""Unit tests for SQLite reminder storage and due time parsing."""
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from storage.reminders import (
    ReminderRecord,
    ReminderStatus,
    ReminderStorage,
    parse_due_time,
)


def test_parse_due_time_relative():
    base = datetime(2026, 10, 10, 12, 0, 0).astimezone()

    # in 10 minutes
    dt1, tz1 = parse_due_time("in 10 minutes", base_now=base)
    assert dt1 == base + timedelta(minutes=10)
    assert len(tz1) > 0

    # in 2 hours
    dt2, _ = parse_due_time("in 2 hours", base_now=base)
    assert dt2 == base + timedelta(hours=2)

    # in 30 seconds
    dt3, _ = parse_due_time("in 30s", base_now=base)
    assert dt3 == base + timedelta(seconds=30)

    # in 1 day
    dt4, _ = parse_due_time("in 1 day", base_now=base)
    assert dt4 == base + timedelta(days=1)


def test_parse_due_time_explicit_and_iso():
    base = datetime(2026, 10, 10, 12, 0, 0).astimezone()

    # ISO string
    iso_str = "2026-10-10T15:30:00"
    dt, _ = parse_due_time(iso_str, base_now=base)
    assert dt.hour == 15
    assert dt.minute == 30

    # Standard date format
    std_str = "2026-10-10 18:00"
    dt2, _ = parse_due_time(std_str, base_now=base)
    assert dt2.hour == 18
    assert dt2.minute == 0


def test_parse_due_time_rejects_ambiguity_and_past():
    base = datetime(2026, 10, 10, 12, 0, 0).astimezone()

    # Ambiguous phrases
    for phrase in ["later", "remind me later", "sometime", "soon", "tonight", "morning"]:
        with pytest.raises(ValueError, match="Ambiguous due time"):
            parse_due_time(phrase, base_now=base)

    # Empty
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_due_time("   ", base_now=base)

    # Past date
    with pytest.raises(ValueError, match="in the past"):
        parse_due_time("2020-01-01 10:00", base_now=base)


def test_storage_initialization_and_persistence(tmp_path: Path):
    db_file = tmp_path / "test_reminders.db"

    # Connection 1: initialize and insert
    storage1 = ReminderStorage(db_path=db_file)
    now = datetime.now().astimezone()
    due = now + timedelta(hours=1)

    rec1 = storage1.create_reminder(
        text="Review deployment logs",
        due_datetime=due,
        timezone_str="IST",
    )
    assert rec1.id.startswith("rem_")
    assert rec1.text == "Review deployment logs"
    assert rec1.status == ReminderStatus.PENDING.value

    # Connection 2: reopen database from file and verify persistence
    storage2 = ReminderStorage(db_path=db_file)
    loaded = storage2.get_reminder(rec1.id)
    assert loaded is not None
    assert loaded.id == rec1.id
    assert loaded.text == "Review deployment logs"
    assert loaded.due_timestamp == rec1.due_timestamp


def test_storage_lifecycle_operations(tmp_path: Path):
    db_file = tmp_path / "lifecycle.db"
    storage = ReminderStorage(db_path=db_file)
    now = datetime.now().astimezone()

    # Create 3 reminders
    r1 = storage.create_reminder("Task 1", now + timedelta(minutes=5), "IST")
    r2 = storage.create_reminder("Task 2", now + timedelta(minutes=10), "IST")
    r3 = storage.create_reminder("Task 3", now + timedelta(minutes=15), "IST")

    # List pending
    pending = storage.list_reminders(status="pending")
    assert len(pending) == 3

    # Cancel Task 2
    assert storage.cancel_reminder(r2.id) is True
    # Cancelling again returns False
    assert storage.cancel_reminder(r2.id) is False

    # Mark Task 1 delivered
    assert storage.mark_delivered(r1.id) is True

    # Check updated lists
    pending_after = storage.list_reminders(status="pending")
    assert len(pending_after) == 1
    assert pending_after[0].id == r3.id

    delivered = storage.list_reminders(status="delivered")
    assert len(delivered) == 1
    assert delivered[0].id == r1.id

    cancelled = storage.list_reminders(status="cancelled")
    assert len(cancelled) == 1
    assert cancelled[0].id == r2.id


def test_storage_due_reminders_query(tmp_path: Path):
    db_file = tmp_path / "due.db"
    storage = ReminderStorage(db_path=db_file)
    now = datetime.now().astimezone()

    # Create one due in the past, one due in 10 minutes
    r_past = storage.create_reminder("Overdue task", now - timedelta(seconds=10), "IST")
    r_future = storage.create_reminder("Future task", now + timedelta(minutes=10), "IST")

    # Check due as of now
    due = storage.get_due_reminders(now.timestamp())
    due_ids = [r.id for r in due]

    assert r_past.id in due_ids
    assert r_future.id not in due_ids
