"""Persistent SQLite storage and time parsing for JARVIS reminders."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


class ReminderStatus(str, Enum):
    """Lifecycle statuses of a reminder."""

    PENDING = "pending"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    MISSED = "missed"


@dataclass
class ReminderRecord:
    """Structured representation of a reminder stored in SQLite."""

    id: str
    text: str
    due_at: str
    due_timestamp: float
    timezone: str
    status: str
    created_at: str
    delivered_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    delivery_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_AMBIGUOUS_KEYWORDS = {
    "later",
    "sometime",
    "soon",
    "morning",
    "afternoon",
    "evening",
    "tonight",
    "someday",
    "eventually",
}


def parse_due_time(
    time_str: str, base_now: Optional[datetime] = None
) -> tuple[datetime, str]:
    """Parse and validate a due time expression into an unambiguous timezone-aware datetime.

    Accepts:
    - Relative offsets: 'in 5 minutes', 'in 2 hours', 'in 30s', 'in 1 day'
    - ISO 8601 strings: '2026-10-10T14:30:00+05:30', '2026-10-10T14:30:00Z', '2026-10-10T14:30:00'
    - Standard explicit formats: 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM'

    Rejects:
    - Ambiguous terms ('later', 'sometime', 'tonight' without specific hour)
    - Non-parseable strings
    - Historical / past timestamps
    """
    raw = time_str.strip()
    if not raw:
        raise ValueError("Due time cannot be empty.")

    now = base_now or datetime.now().astimezone()
    local_tz = now.tzinfo
    raw_offset = now.strftime("%z")
    formatted_offset = f"UTC{raw_offset[:3]}:{raw_offset[3:]}" if raw_offset else ""
    tz_label = (
        f"{now.tzname() or 'Local'} ({formatted_offset})"
        if formatted_offset
        else (now.tzname() or "Local")
    )

    lowered = raw.lower()

    # Reject inherently ambiguous expressions
    if lowered in _AMBIGUOUS_KEYWORDS or lowered.startswith("remind me later"):
        raise ValueError(
            f"Ambiguous due time '{raw}'. Please specify an exact time (e.g. '2026-10-10 15:00') "
            "or a clear relative duration (e.g. 'in 15 minutes')."
        )

    # 1. Match relative durations (e.g. 'in 10 minutes', 'in 2 hours', 'in 30 seconds')
    m_rel = re.match(
        r"^in\s+(\d+(?:\.\d+)?)\s*(s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:our?s?)?|d(?:ays?)?)$",
        lowered,
    )
    if m_rel:
        qty = float(m_rel.group(1))
        unit = m_rel.group(2)
        if qty <= 0:
            raise ValueError("Relative reminder duration must be greater than zero.")

        if unit.startswith("s"):
            delta = timedelta(seconds=qty)
        elif unit.startswith("m"):
            delta = timedelta(minutes=qty)
        elif unit.startswith("h"):
            delta = timedelta(hours=qty)
        elif unit.startswith("d"):
            delta = timedelta(days=qty)
        else:
            raise ValueError(f"Unrecognized time unit '{unit}'.")

        due_dt = now + delta
        return due_dt, tz_label

    # 2. Try ISO 8601 parsing
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
        if dt < now - timedelta(seconds=5):
            raise ValueError(
                f"Due time '{raw}' is in the past. Reminders must be scheduled for a future time."
            )
        return dt, tz_label
    except ValueError as exc:
        if "in the past" in str(exc):
            raise

    # 3. Try standard explicit formats
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]:
        try:
            dt = datetime.strptime(raw, fmt)
            dt = dt.replace(tzinfo=local_tz)
            if dt < now - timedelta(seconds=5):
                raise ValueError(
                    f"Due time '{raw}' is in the past. Reminders must be scheduled for a future time."
                )
            return dt, tz_label
        except ValueError as exc:
            if "in the past" in str(exc):
                raise

    raise ValueError(
        f"Unable to parse due time '{raw}'. Expected ISO format (e.g. 'YYYY-MM-DDTHH:MM:SS') "
        "or relative duration (e.g. 'in 10 minutes')."
    )


class ReminderStorage:
    """Thread-safe SQLite storage for JARVIS reminders."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = (db_path or settings.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database schema and performance indices if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    due_timestamp REAL NOT NULL,
                    timezone TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    cancelled_at TEXT,
                    delivery_error TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_status_due
                ON reminders (status, due_timestamp);
            """)

    def create_reminder(
        self,
        text: str,
        due_datetime: datetime,
        timezone_str: str,
    ) -> ReminderRecord:
        """Create a new pending reminder with a unique identifier."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Reminder text cannot be empty.")

        reminder_id = f"rem_{uuid.uuid4().hex[:8]}"
        due_at_iso = due_datetime.isoformat()
        due_epoch = due_datetime.timestamp()
        created_at_iso = datetime.now().astimezone().isoformat()
        status = ReminderStatus.PENDING.value

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reminders (
                    id, text, due_at, due_timestamp, timezone,
                    status, created_at, delivered_at, cancelled_at, delivery_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL);
                """,
                (
                    reminder_id,
                    clean_text,
                    due_at_iso,
                    due_epoch,
                    timezone_str,
                    status,
                    created_at_iso,
                ),
            )

        return ReminderRecord(
            id=reminder_id,
            text=clean_text,
            due_at=due_at_iso,
            due_timestamp=due_epoch,
            timezone=timezone_str,
            status=status,
            created_at=created_at_iso,
        )

    def list_reminders(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[ReminderRecord]:
        """List reminders, optionally filtered by status ('pending', 'delivered', etc.)."""
        with self._get_connection() as conn:
            if status and status.lower() != "all":
                cursor = conn.execute(
                    """
                    SELECT * FROM reminders
                    WHERE status = ?
                    ORDER BY due_timestamp ASC
                    LIMIT ?;
                    """,
                    (status.lower(), limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM reminders
                    ORDER BY due_timestamp ASC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            return [ReminderRecord(**dict(row)) for row in rows]

    def get_reminder(self, reminder_id: str) -> Optional[ReminderRecord]:
        """Retrieve a specific reminder by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM reminders WHERE id = ?;", (reminder_id.strip(),)
            )
            row = cursor.fetchone()
            if row:
                return ReminderRecord(**dict(row))
            return None

    def cancel_reminder(self, reminder_id: str) -> bool:
        """Cancel a pending reminder. Returns True if cancelled, False otherwise."""
        cancelled_at_iso = datetime.now().astimezone().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE reminders
                SET status = ?, cancelled_at = ?
                WHERE id = ? AND status = ?;
                """,
                (
                    ReminderStatus.CANCELLED.value,
                    cancelled_at_iso,
                    reminder_id.strip(),
                    ReminderStatus.PENDING.value,
                ),
            )
            return cursor.rowcount > 0

    def get_due_reminders(self, as_of_timestamp: float) -> List[ReminderRecord]:
        """Retrieve all pending reminders whose due time has arrived or passed."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status = ? AND due_timestamp <= ?
                ORDER BY due_timestamp ASC;
                """,
                (ReminderStatus.PENDING.value, as_of_timestamp),
            )
            rows = cursor.fetchall()
            return [ReminderRecord(**dict(row)) for row in rows]

    def mark_delivered(
        self,
        reminder_id: str,
        delivered_at: Optional[datetime] = None,
        error: Optional[str] = None,
        status: str = ReminderStatus.DELIVERED.value,
    ) -> bool:
        """Mark a reminder as delivered (or failed/missed) with execution timestamp and optional error note."""
        delivered_iso = (
            delivered_at or datetime.now().astimezone()
        ).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE reminders
                SET status = ?, delivered_at = ?, delivery_error = ?
                WHERE id = ?;
                """,
                (status, delivered_iso, error, reminder_id.strip()),
            )
            return cursor.rowcount > 0


# Global shared instance
default_reminder_storage = ReminderStorage()
