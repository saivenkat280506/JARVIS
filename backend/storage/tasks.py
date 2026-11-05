"""Persistent SQLite Task Lifecycle Storage and Crash Recovery for JARVIS.

Manages durable task records in data/jarvis.db with:
- Formal state machine: queued -> running -> waiting_for_approval -> [succeeded | failed | cancelled | interrupted]
- Terminal state immutability.
- Explicit retry tracking (creating new linked execution attempts).
- Startup recovery invalidation of abandoned active tasks.
- Concurrency-safe atomic state transitions with threading.RLock and SQLite conditional updates.
- Strict input validation and secret/token redaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Explicit lifecycle statuses of a persistent task."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


# Terminal states cannot transition to any other state
TERMINAL_STATUSES: Set[TaskStatus] = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}

# Permitted state transition map
VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.QUEUED: {
        TaskStatus.RUNNING,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    },
    TaskStatus.RUNNING: {
        TaskStatus.WAITING_FOR_APPROVAL,
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    },
    TaskStatus.WAITING_FOR_APPROVAL: {
        TaskStatus.RUNNING,
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    },
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.INTERRUPTED: set(),
}


class TaskStorageError(Exception):
    """Base exception for task storage operations."""


class TaskNotFoundError(TaskStorageError):
    """Raised when a requested task does not exist."""


class InvalidTaskStateTransitionError(TaskStorageError):
    """Raised when an illegal state transition or modification to a terminal task is attempted."""


@dataclass
class TaskRecord:
    """Structured representation of a persistent task."""

    id: str
    task_type: str
    status: TaskStatus
    description: str
    conversation_id: Optional[str]
    created_at: str
    created_timestamp: float
    started_at: Optional[str]
    updated_at: str
    completed_at: Optional[str]
    status_message: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Return True if task has reached an immutable terminal state."""
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a clean dictionary."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else str(self.status)
        return d


def sanitize_task_text(text: Optional[str]) -> str:
    """Scrub raw approval tokens, passwords, bearer tokens, and secrets from task text."""
    if not text:
        return ""
    scrubbed = re.sub(r"\btok_[0-9a-fA-F]{16,64}\b", "[REDACTED_TOKEN]", text)
    scrubbed = re.sub(r"\bbearer[_\-\w]+\b", "[REDACTED_BEARER]", scrubbed, flags=re.IGNORECASE)
    pattern = r"((?:password|passwd|pwd|secret|token|bearer|key|credential|auth)\w*(?:[^\w\s]*\s*(?:field|input|value|box))?\s*(?:[:=]|is|with|to|\s)\s*)(?!\[REDACTED)([\w!@#$%^&*()_+=\-\[\]{}|;:,.<>?/~`]+)"
    scrubbed = re.sub(pattern, r"\1[REDACTED]", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'("(?:password|secret|auth_token|token|key|credential)"\s*:\s*)"[^"]+"', r'\1"[REDACTED]"', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'("approval_token"\s*:\s*)"[^"]+"', r'\1"[REDACTED_TOKEN]"', scrubbed, flags=re.IGNORECASE)
    return scrubbed


def sanitize_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively scrub sensitive keys and token values from metadata dictionary."""
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(s in k_lower for s in ["pass", "secret", "token", "key", "auth", "credential"]):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_metadata(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_metadata(item) if isinstance(item, dict) else sanitize_task_text(item) if isinstance(item, str) else item
                for item in v
            ]
        elif isinstance(v, str):
            sanitized[k] = sanitize_task_text(v)
        else:
            sanitized[k] = v
    return sanitized


class TaskStorage:
    """Thread-safe, transaction-safe SQLite task lifecycle manager with crash recovery."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        auto_recover: bool = True,
    ) -> None:
        self.db_path = (db_path or settings.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
        if auto_recover:
            self.recover_interrupted_tasks()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tasks table and indexes if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        description TEXT NOT NULL,
                        conversation_id TEXT,
                        created_at TEXT NOT NULL,
                        created_timestamp REAL NOT NULL,
                        started_at TEXT,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        status_message TEXT NOT NULL DEFAULT '',
                        error_message TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_status
                    ON tasks (status);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_conversation
                    ON tasks (conversation_id);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON tasks (created_timestamp DESC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_type_status
                    ON tasks (task_type, status);
                """)
        except sqlite3.Error as exc:
            logger.error("Failed to initialize tasks schema in %s: %s", self.db_path, exc)
            raise TaskStorageError(f"Database error initializing tasks schema: {exc}") from exc

    def _row_to_record(self, row: sqlite3.Row) -> TaskRecord:
        """Convert a database row into a validated TaskRecord."""
        raw_status = row["status"]
        status_enum = (
            TaskStatus(raw_status)
            if raw_status in TaskStatus._value2member_map_
            else TaskStatus.FAILED
        )
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return TaskRecord(
            id=row["id"],
            task_type=row["task_type"],
            status=status_enum,
            description=row["description"],
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            created_timestamp=row["created_timestamp"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            status_message=row["status_message"] or "",
            error_message=row["error_message"],
            metadata=meta,
        )

    def create_task(
        self,
        task_type: str,
        description: str,
        conversation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> TaskRecord:
        """Persist a new task in QUEUED or RUNNING status with sanitized fields."""
        if status in TERMINAL_STATUSES:
            raise InvalidTaskStateTransitionError(
                f"Cannot create a new task directly in terminal status '{status.value}'."
            )

        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        now_iso = datetime.now().astimezone().isoformat()
        clean_desc = sanitize_task_text(description)[:250]
        clean_meta = sanitize_metadata(metadata or {})
        meta_json = json.dumps(clean_meta)

        started_at = now_iso if status == TaskStatus.RUNNING else None

        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO tasks (
                            id, task_type, status, description, conversation_id,
                            created_at, created_timestamp, started_at, updated_at,
                            completed_at, status_message, error_message, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, '', NULL, ?);
                        """,
                        (
                            tid,
                            task_type,
                            status.value,
                            clean_desc,
                            conversation_id,
                            now_iso,
                            now,
                            started_at,
                            now_iso,
                            meta_json,
                        ),
                    )
                logger.info("Created persistent task %s (type=%s, status=%s)", tid, task_type, status.value)
                task = self.get_task(tid)
                if not task:
                    raise TaskStorageError(f"Task {tid} created but could not be retrieved.")
                return task
            except sqlite3.Error as exc:
                logger.error("Failed to create task %s: %s", tid, exc)
                raise TaskStorageError(f"Database error creating task: {exc}") from exc

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve a task record by unique ID."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM tasks WHERE id = ?;",
                        (task_id,),
                    ).fetchone()
                    if not row:
                        return None
                    return self._row_to_record(row)
            except sqlite3.Error as exc:
                logger.error("Failed to retrieve task %s: %s", task_id, exc)
                raise TaskStorageError(f"Database error retrieving task: {exc}") from exc

    def update_task_status(
        self,
        task_id: str,
        new_status: TaskStatus,
        status_message: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        expected_status: Optional[TaskStatus] = None,
    ) -> TaskRecord:
        """Atomically transition a task to a new status with validation and terminal-state protection."""
        now_iso = datetime.now().astimezone().isoformat()
        clean_msg = sanitize_task_text(status_message)[:500] if status_message is not None else None
        clean_err = sanitize_task_text(error_message)[:1000] if error_message is not None else None

        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM tasks WHERE id = ?;",
                        (task_id,),
                    ).fetchone()
                    if not row:
                        raise TaskNotFoundError(f"Task '{task_id}' not found.")

                    current_record = self._row_to_record(row)
                    current_status = current_record.status

                    # Check expected status if caller specified one
                    if expected_status is not None and current_status != expected_status:
                        raise InvalidTaskStateTransitionError(
                            f"Task '{task_id}' status mismatch: expected '{expected_status.value}', currently '{current_status.value}'."
                        )

                    # Terminal-state immutability
                    if current_status in TERMINAL_STATUSES:
                        raise InvalidTaskStateTransitionError(
                            f"Task '{task_id}' is in terminal status '{current_status.value}' and cannot transition to '{new_status.value}'."
                        )

                    # Validate legal transition
                    allowed = VALID_TRANSITIONS.get(current_status, set())
                    if new_status not in allowed:
                        raise InvalidTaskStateTransitionError(
                            f"Illegal task transition from '{current_status.value}' to '{new_status.value}' for task '{task_id}'."
                        )

                    # Merge metadata
                    updated_meta = dict(current_record.metadata)
                    if metadata_updates:
                        updated_meta.update(sanitize_metadata(metadata_updates))
                    meta_json = json.dumps(updated_meta)

                    # Started timestamp set on entering running
                    started_at = (
                        current_record.started_at
                        if current_record.started_at
                        else (now_iso if new_status == TaskStatus.RUNNING else None)
                    )

                    # Completed timestamp set on entering terminal state
                    completed_at = (
                        now_iso if new_status in TERMINAL_STATUSES else current_record.completed_at
                    )

                    # Use previous values if not explicitly provided
                    final_msg = clean_msg if clean_msg is not None else current_record.status_message
                    final_err = clean_err if clean_err is not None else current_record.error_message

                    cursor = conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?,
                            updated_at = ?,
                            started_at = COALESCE(started_at, ?),
                            completed_at = ?,
                            status_message = ?,
                            error_message = ?,
                            metadata_json = ?
                        WHERE id = ? AND status = ?;
                        """,
                        (
                            new_status.value,
                            now_iso,
                            started_at,
                            completed_at,
                            final_msg,
                            final_err,
                            meta_json,
                            task_id,
                            current_status.value,
                        ),
                    )

                    if cursor.rowcount == 0:
                        raise InvalidTaskStateTransitionError(
                            f"Concurrent modification detected: Task '{task_id}' status changed from '{current_status.value}' concurrently."
                        )

                    conn.commit()
                    logger.info("Task %s transitioned: %s -> %s", task_id, current_status.value, new_status.value)
                return self.get_task(task_id)  # type: ignore[return-value]

            except sqlite3.Error as exc:
                logger.error("Failed to update status for task %s: %s", task_id, exc)
                raise TaskStorageError(f"Database error updating task status: {exc}") from exc

    def cancel_task(self, task_id: str, reason: str = "Cancelled by user") -> TaskRecord:
        """Cancel an active task."""
        clean_reason = sanitize_task_text(reason)
        return self.update_task_status(
            task_id=task_id,
            new_status=TaskStatus.CANCELLED,
            status_message=f"Cancelled: {clean_reason}",
        )

    def create_retry_task(
        self,
        parent_task_id: str,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        """Explicitly create a new execution attempt linked to an immutable failed/cancelled/interrupted parent task."""
        parent = self.get_task(parent_task_id)
        if not parent:
            raise TaskNotFoundError(f"Parent task '{parent_task_id}' not found.")

        if parent.status not in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.INTERRUPTED}:
            raise InvalidTaskStateTransitionError(
                f"Cannot retry task '{parent_task_id}' because it is in status '{parent.status.value}'. Only failed, cancelled, or interrupted tasks can be retried."
            )

        retry_attempt = parent.metadata.get("retry_attempt", 0) + 1
        meta = dict(parent.metadata)
        meta["retry_of_task_id"] = parent_task_id
        meta["retry_attempt"] = retry_attempt
        if metadata_updates:
            meta.update(metadata_updates)

        new_desc = description or f"Retry #{retry_attempt} of: {parent.description}"
        new_type = task_type or parent.task_type

        return self.create_task(
            task_type=new_type,
            description=new_desc,
            conversation_id=parent.conversation_id,
            metadata=meta,
            status=TaskStatus.QUEUED,
        )

    def list_tasks(
        self,
        status: Optional[Union[TaskStatus, str]] = None,
        conversation_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        """List tasks with optional filters, sorted by creation timestamp descending."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    query = "SELECT * FROM tasks WHERE 1=1"
                    params: List[Any] = []

                    if status is not None:
                        s_val = status.value if isinstance(status, TaskStatus) else str(status)
                        query += " AND status = ?"
                        params.append(s_val)

                    if conversation_id is not None:
                        query += " AND conversation_id = ?"
                        params.append(conversation_id)

                    if task_type is not None:
                        query += " AND task_type = ?"
                        params.append(task_type)

                    query += " ORDER BY created_timestamp DESC LIMIT ?;"
                    params.append(max(1, limit))

                    rows = conn.execute(query, params).fetchall()
                    return [self._row_to_record(r) for r in rows]
            except sqlite3.Error as exc:
                logger.error("Failed to list tasks: %s", exc)
                raise TaskStorageError(f"Database error listing tasks: {exc}") from exc

    def recover_interrupted_tasks(self) -> List[TaskRecord]:
        """Crash recovery: inspect tasks left in active states by a previous process and mark them 'interrupted'.

        Never automatically executes actions during recovery.
        Explicitly records uncertainty regarding execution completion.
        """
        now_iso = datetime.now().astimezone().isoformat()
        recovered_tasks: List[TaskRecord] = []
        with self._lock:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        """
                        SELECT * FROM tasks
                        WHERE status IN ('queued', 'running', 'waiting_for_approval');
                        """
                    ).fetchall()

                    for r in rows:
                        task_id = r["id"]
                        prev_status = r["status"]
                        meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                        meta["recovered_from_status"] = prev_status
                        meta["interrupted_at"] = now_iso

                        if prev_status == TaskStatus.WAITING_FOR_APPROVAL.value:
                            msg = (
                                "Task interrupted by process shutdown while awaiting user approval. "
                                "Pending approval tokens were invalidated on startup; external action was NOT executed."
                            )
                        elif prev_status == TaskStatus.RUNNING.value:
                            msg = (
                                "Task execution was interrupted by process shutdown. "
                                "Completion status is unknown; external action was NOT automatically re-executed."
                            )
                        else:  # queued
                            msg = (
                                "Task was queued when process terminated. "
                                "Execution interrupted before start."
                            )

                        conn.execute(
                            """
                            UPDATE tasks
                            SET status = 'interrupted',
                                updated_at = ?,
                                completed_at = ?,
                                status_message = ?,
                                metadata_json = ?
                            WHERE id = ? AND status = ?;
                            """,
                            (
                                now_iso,
                                now_iso,
                                msg,
                                json.dumps(meta),
                                task_id,
                                prev_status,
                            ),
                        )

                    conn.commit()

                for r in rows:
                    task_rec = self.get_task(r["id"])
                    if task_rec:
                        recovered_tasks.append(task_rec)

                if recovered_tasks:
                    logger.warning(
                        "Startup recovery: marked %d active task(s) as interrupted",
                        len(recovered_tasks),
                    )
                return recovered_tasks
            except sqlite3.Error as exc:
                logger.error("Failed to recover interrupted tasks: %s", exc)
                raise TaskStorageError(f"Database error during crash recovery: {exc}") from exc


# Default global instance
default_task_storage = TaskStorage(auto_recover=True)
