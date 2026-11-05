"""Reminder tools for JARVIS."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

from storage.reminders import default_reminder_storage, parse_due_time


class CreateReminderArgs(BaseModel):
    """Arguments for creating a persistent reminder."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The content of the reminder (e.g. 'Review quarterly system architecture')",
    )
    due_time: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Due timestamp or duration, e.g. 'in 10 minutes' or '2026-10-10 15:30'",
    )
    model_config = ConfigDict(extra="forbid")


class ListRemindersArgs(BaseModel):
    """Arguments for querying reminders."""

    status: Optional[str] = Field(
        default="pending",
        description="Status filter: defaults to 'pending' for active reminders. Use 'all' only when explicitly asked for all historical reminders.",
    )
    model_config = ConfigDict(extra="forbid")


class CancelReminderArgs(BaseModel):
    """Arguments for cancelling a reminder."""

    reminder_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique identifier of the reminder to cancel (e.g. 'rem_a1b2c3d4')",
    )
    model_config = ConfigDict(extra="forbid")


def create_reminder(text: str, due_time: str) -> Dict[str, Any]:
    """Tool handler for creating a new scheduled reminder."""
    try:
        due_dt, tz_label = parse_due_time(due_time)
    except ValueError as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    try:
        record = default_reminder_storage.create_reminder(
            text=text,
            due_datetime=due_dt,
            timezone_str=tz_label,
        )
        return {
            "success": True,
            "reminder": record.to_dict(),
            "message": (
                f"Reminder created with ID '{record.id}'. "
                f"Scheduled for {record.due_at} ({record.timezone})."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to save reminder to SQLite database: {exc}",
        }


def list_reminders(status: Optional[str] = "pending") -> Dict[str, Any]:
    """Tool handler for listing stored reminders."""
    try:
        effective_status = status or "pending"
        records = default_reminder_storage.list_reminders(status=effective_status, limit=10)
        if effective_status == "all":
            pending = [r for r in records if r.status == "pending"]
            if pending:
                records = pending
            else:
                non_cancelled = [r for r in records if r.status != "cancelled"]
                records = non_cancelled if non_cancelled else records[:3]

        return {
            "success": True,
            "count": len(records),
            "status_filter": effective_status,
            "reminders": [
                {
                    "id": r.id,
                    "text": r.text,
                    "due_at": r.due_at,
                    "status": r.status,
                }
                for r in records
            ],
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to query reminders: {exc}",
        }


def cancel_reminder(reminder_id: str) -> Dict[str, Any]:
    """Tool handler for cancelling an existing pending reminder."""
    try:
        success = default_reminder_storage.cancel_reminder(reminder_id)
        if success:
            return {
                "success": True,
                "reminder_id": reminder_id,
                "message": f"Reminder '{reminder_id}' was cancelled.",
            }
        else:
            return {
                "success": False,
                "error": f"Reminder '{reminder_id}' was not found or is not in 'pending' status.",
            }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to cancel reminder: {exc}",
        }
