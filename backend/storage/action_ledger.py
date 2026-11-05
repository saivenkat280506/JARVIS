"""Persistent Consequential Action Idempotency Ledger and Outcome Reconciliation for JARVIS.

Manages durable action execution records in data/jarvis.db with:
- Formal action identity: task_id, step_name, tool_name, arguments_hash, attempt_number.
- Stable idempotency keys across retries of the same logical action.
- Separation between authorization tokens and idempotency keys.
- State machine: prepared -> awaiting_approval -> approved -> executing -> [succeeded | failed | outcome_unknown | cancelled].
- Tool-specific retry safety classification: READ_ONLY, NATURALLY_IDEMPOTENT, EXTERNAL_IDEMPOTENT, NON_IDEMPOTENT.
- Safe crash recovery: active non-terminal actions marked outcome_unknown or cancelled without automatic re-execution.
- Read-only outcome reconciliation for browser operations and external effects.
- Concurrency-safe atomic compare-and-swap transitions and secret redaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from config.settings import settings
from storage.tasks import sanitize_metadata, sanitize_task_text

logger = logging.getLogger(__name__)


class ActionLedgerStatus(str, Enum):
    """Lifecycle status of an action execution attempt."""

    PREPARED = "prepared"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CANCELLED = "cancelled"


TERMINAL_LEDGER_STATUSES: Set[ActionLedgerStatus] = {
    ActionLedgerStatus.SUCCEEDED,
    ActionLedgerStatus.FAILED,
    ActionLedgerStatus.OUTCOME_UNKNOWN,
    ActionLedgerStatus.CANCELLED,
}

VALID_LEDGER_TRANSITIONS: Dict[ActionLedgerStatus, Set[ActionLedgerStatus]] = {
    ActionLedgerStatus.PREPARED: {
        ActionLedgerStatus.AWAITING_APPROVAL,
        ActionLedgerStatus.APPROVED,
        ActionLedgerStatus.EXECUTING,
        ActionLedgerStatus.CANCELLED,
    },
    ActionLedgerStatus.AWAITING_APPROVAL: {
        ActionLedgerStatus.APPROVED,
        ActionLedgerStatus.CANCELLED,
        ActionLedgerStatus.OUTCOME_UNKNOWN,
    },
    ActionLedgerStatus.APPROVED: {
        ActionLedgerStatus.EXECUTING,
        ActionLedgerStatus.CANCELLED,
        ActionLedgerStatus.OUTCOME_UNKNOWN,
    },
    ActionLedgerStatus.EXECUTING: {
        ActionLedgerStatus.SUCCEEDED,
        ActionLedgerStatus.FAILED,
        ActionLedgerStatus.OUTCOME_UNKNOWN,
    },
    # outcome_unknown can only transition via explicit reconciliation with conclusive proof
    ActionLedgerStatus.OUTCOME_UNKNOWN: {
        ActionLedgerStatus.SUCCEEDED,
        ActionLedgerStatus.FAILED,
    },
    ActionLedgerStatus.SUCCEEDED: set(),
    ActionLedgerStatus.FAILED: set(),
    ActionLedgerStatus.CANCELLED: set(),
}


class RetrySafety(str, Enum):
    """Classification of tool operations by retry safety."""

    READ_ONLY = "read_only"
    NATURALLY_IDEMPOTENT = "naturally_idempotent"
    EXTERNAL_IDEMPOTENT = "external_idempotent"
    NON_IDEMPOTENT = "non_idempotent"


class ActionLedgerError(Exception):
    """Base exception for action ledger operations."""


class ActionLedgerNotFoundError(ActionLedgerError):
    """Raised when an action attempt does not exist."""


class ActionLedgerConflictError(ActionLedgerError):
    """Raised when an in-flight execution or conflicting duplicate is detected."""


class InvalidActionStateTransitionError(ActionLedgerError):
    """Raised when an illegal transition is attempted in the ledger."""


class ActionUncertaintyError(ActionLedgerError):
    """Raised when an action has outcome_unknown and cannot be blindly retried."""


def generate_idempotency_key(
    task_id: str,
    step_name: str,
    client_key: Optional[str] = None,
) -> str:
    """Produce a stable idempotency key identifying a logical action attempt."""
    if client_key:
        clean_key = re.sub(r"[^\w\-_.:]", "_", client_key.strip())[:64]
        return f"{task_id}:{clean_key}"
    clean_step = re.sub(r"[^\w\-_.:]", "_", step_name.strip())[:64]
    return f"{task_id}:{clean_step}"


def normalize_and_hash_arguments(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Normalize arguments by stripping transient fields and compute canonical SHA-256."""
    canonical = {}
    for k, v in sorted(arguments.items()):
        # Exclude ephemeral authorization or dispatch tokens from argument hash
        if k in ("approval_token", "user_confirmed", "action_id"):
            continue
        canonical[k] = v
    sanitized = sanitize_metadata(canonical)
    encoded = json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sanitized, hashlib.sha256(encoded).hexdigest()


@dataclass
class ActionAttemptRecord:
    """Structured record of a single action attempt in the ledger."""

    id: str
    idempotency_key: str
    task_id: str
    step_name: str
    tool_name: str
    arguments_hash: str
    attempt_number: int
    status: ActionLedgerStatus
    retry_safety: RetrySafety
    action_id: Optional[str]
    created_at: str
    created_timestamp: float
    started_at: Optional[str]
    completed_at: Optional[str]
    status_message: str = ""
    error_message: Optional[str] = None
    reconciliation_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if attempt is in an immutable state."""
        return self.status in TERMINAL_LEDGER_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, ActionLedgerStatus) else str(self.status)
        d["retry_safety"] = self.retry_safety.value if isinstance(self.retry_safety, RetrySafety) else str(self.retry_safety)
        return d


class ActionLedger:
    """Durable, thread-safe SQLite ledger tracking action execution attempts and outcome reconciliation."""

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
            self.recover_abandoned_actions()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize the action_ledger table and indexes in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS action_ledger (
                        id TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments_hash TEXT NOT NULL,
                        attempt_number INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL,
                        retry_safety TEXT NOT NULL,
                        action_id TEXT,
                        created_at TEXT NOT NULL,
                        created_timestamp REAL NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        status_message TEXT NOT NULL DEFAULT '',
                        error_message TEXT,
                        reconciliation_notes TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE (idempotency_key, attempt_number)
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ledger_idempotency_key
                    ON action_ledger (idempotency_key);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ledger_task_id
                    ON action_ledger (task_id);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ledger_status
                    ON action_ledger (status);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ledger_tool_status
                    ON action_ledger (tool_name, status);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ledger_action_id
                    ON action_ledger (action_id);
                """)
        except sqlite3.Error as exc:
            logger.error("Failed to initialize action_ledger in %s: %s", self.db_path, exc)
            raise ActionLedgerError(f"Database error initializing action_ledger: {exc}") from exc

    def _row_to_record(self, row: sqlite3.Row) -> ActionAttemptRecord:
        raw_status = row["status"]
        status_enum = (
            ActionLedgerStatus(raw_status)
            if raw_status in ActionLedgerStatus._value2member_map_
            else ActionLedgerStatus.OUTCOME_UNKNOWN
        )
        raw_safety = row["retry_safety"]
        safety_enum = (
            RetrySafety(raw_safety)
            if raw_safety in RetrySafety._value2member_map_
            else RetrySafety.NON_IDEMPOTENT
        )
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

        return ActionAttemptRecord(
            id=row["id"],
            idempotency_key=row["idempotency_key"],
            task_id=row["task_id"],
            step_name=row["step_name"],
            tool_name=row["tool_name"],
            arguments_hash=row["arguments_hash"],
            attempt_number=row["attempt_number"],
            status=status_enum,
            retry_safety=safety_enum,
            action_id=row["action_id"],
            created_at=row["created_at"],
            created_timestamp=row["created_timestamp"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status_message=row["status_message"] or "",
            error_message=row["error_message"],
            reconciliation_notes=row["reconciliation_notes"],
            metadata=meta,
        )

    def prepare_action(
        self,
        task_id: str,
        step_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        retry_safety: RetrySafety = RetrySafety.NON_IDEMPOTENT,
        action_id: Optional[str] = None,
        client_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initial_status: ActionLedgerStatus = ActionLedgerStatus.PREPARED,
    ) -> Tuple[ActionAttemptRecord, bool]:
        """Prepare an action attempt in the ledger.

        Returns:
            Tuple of (record, is_cached_existing_success):
            - If an existing attempt for this logical action has already SUCCEEDED, returns
              (existing_record, True) allowing caller to reuse result without duplicate execution.
            - If an existing attempt is EXECUTING, raises ActionLedgerConflictError.
            - If an existing attempt is OUTCOME_UNKNOWN, raises ActionUncertaintyError.
            - If an existing attempt FAILED, creates a new attempt record with incremented attempt_number.
            - Otherwise creates a fresh attempt #1.
        """
        idempotency_key = generate_idempotency_key(task_id, step_name, client_key)
        _, args_hash = normalize_and_hash_arguments(arguments)
        clean_meta = sanitize_metadata(metadata or {})
        now = time.time()
        now_iso = datetime.now().astimezone().isoformat()

        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Check latest attempt for this idempotency key
                    row = conn.execute(
                        """
                        SELECT * FROM action_ledger
                        WHERE idempotency_key = ?
                        ORDER BY attempt_number DESC LIMIT 1;
                        """,
                        (idempotency_key,),
                    ).fetchone()

                    if row:
                        latest = self._row_to_record(row)

                        # 1. Existing attempt already succeeded -> Deduplicate!
                        if latest.status == ActionLedgerStatus.SUCCEEDED:
                            logger.info(
                                "Idempotent deduplication: Action '%s' already succeeded in attempt #%d",
                                idempotency_key,
                                latest.attempt_number,
                            )
                            return latest, True

                        # 2. Existing attempt currently executing -> Concurrent duplicate!
                        if latest.status == ActionLedgerStatus.EXECUTING:
                            raise ActionLedgerConflictError(
                                f"Action '{idempotency_key}' is already executing in attempt #{latest.attempt_number}."
                            )

                        # 3. Existing attempt outcome is unknown -> Uncertain outcome must not blindly retry!
                        if latest.status == ActionLedgerStatus.OUTCOME_UNKNOWN:
                            raise ActionUncertaintyError(
                                f"Action '{idempotency_key}' attempt #{latest.attempt_number} has unknown outcome; "
                                "cannot retry without reconciliation or explicit user decision."
                            )

                        # 4. Existing attempt is awaiting approval or approved -> Transition attempt directly!
                        if latest.status == ActionLedgerStatus.AWAITING_APPROVAL:
                            if initial_status == ActionLedgerStatus.APPROVED:
                                merged_meta = {**latest.metadata, **clean_meta}
                                conn.execute(
                                    """
                                    UPDATE action_ledger
                                    SET status = ?, metadata_json = ?
                                    WHERE id = ? AND status = ?;
                                    """,
                                    (
                                        ActionLedgerStatus.APPROVED.value,
                                        json.dumps(merged_meta),
                                        latest.id,
                                        ActionLedgerStatus.AWAITING_APPROVAL.value,
                                    ),
                                )
                                conn.commit()
                                updated_rec = self.get_attempt(latest.id)
                                logger.info("Transitioned attempt %s: awaiting_approval -> approved", latest.id)
                                return updated_rec, False
                            elif initial_status == ActionLedgerStatus.AWAITING_APPROVAL:
                                return latest, False
                            elif initial_status == ActionLedgerStatus.CANCELLED:
                                conn.execute(
                                    """
                                    UPDATE action_ledger
                                    SET status = ?, status_message = ?, completed_at = ?
                                    WHERE id = ? AND status = ?;
                                    """,
                                    (
                                        ActionLedgerStatus.CANCELLED.value,
                                        "Action cancelled before approval",
                                        now_iso,
                                        latest.id,
                                        ActionLedgerStatus.AWAITING_APPROVAL.value,
                                    ),
                                )
                                conn.commit()
                                updated_rec = self.get_attempt(latest.id)
                                return updated_rec, False

                        if latest.status == ActionLedgerStatus.APPROVED:
                            if initial_status == ActionLedgerStatus.EXECUTING:
                                conn.execute(
                                    """
                                    UPDATE action_ledger
                                    SET status = ?, started_at = COALESCE(started_at, ?)
                                    WHERE id = ? AND status = ?;
                                    """,
                                    (
                                        ActionLedgerStatus.EXECUTING.value,
                                        now_iso,
                                        latest.id,
                                        ActionLedgerStatus.APPROVED.value,
                                    ),
                                )
                                conn.commit()
                                updated_rec = self.get_attempt(latest.id)
                                return updated_rec, False
                            elif initial_status == ActionLedgerStatus.APPROVED:
                                return latest, False

                        # 5. Existing attempt was cancelled or failed -> Create next attempt number
                        next_attempt = latest.attempt_number + 1
                    else:
                        next_attempt = 1

                    attempt_id = f"att_{uuid.uuid4().hex[:12]}"
                    started_at = now_iso if initial_status == ActionLedgerStatus.EXECUTING else None

                    conn.execute(
                        """
                        INSERT INTO action_ledger (
                            id, idempotency_key, task_id, step_name, tool_name,
                            arguments_hash, attempt_number, status, retry_safety,
                            action_id, created_at, created_timestamp, started_at,
                            completed_at, status_message, error_message, reconciliation_notes,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, '', NULL, NULL, ?);
                        """,
                        (
                            attempt_id,
                            idempotency_key,
                            task_id,
                            step_name,
                            tool_name,
                            args_hash,
                            next_attempt,
                            initial_status.value,
                            retry_safety.value,
                            action_id,
                            now_iso,
                            now,
                            started_at,
                            json.dumps(clean_meta),
                        ),
                    )
                    conn.commit()

                rec = self.get_attempt(attempt_id)
                if not rec:
                    raise ActionLedgerError(f"Attempt '{attempt_id}' created but could not be read.")
                logger.info(
                    "Prepared action attempt '%s' (key=%s, attempt=%d, status=%s)",
                    attempt_id,
                    idempotency_key,
                    next_attempt,
                    initial_status.value,
                )
                return rec, False

            except sqlite3.IntegrityError as ie:
                logger.warning("Concurrent conflict preparing action %s: %s", idempotency_key, ie)
                raise ActionLedgerConflictError(
                    f"Concurrent conflict creating action attempt for key '{idempotency_key}': {ie}"
                ) from ie
            except sqlite3.Error as exc:
                logger.error("Database error preparing action: %s", exc)
                raise ActionLedgerError(f"Database error preparing action: {exc}") from exc

    def get_attempt(self, attempt_id: str) -> Optional[ActionAttemptRecord]:
        """Retrieve an action attempt by unique attempt ID."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM action_ledger WHERE id = ?;",
                        (attempt_id,),
                    ).fetchone()
                    if not row:
                        return None
                    return self._row_to_record(row)
            except sqlite3.Error as exc:
                logger.error("Database error getting attempt %s: %s", attempt_id, exc)
                raise ActionLedgerError(f"Database error retrieving attempt: {exc}") from exc

    def get_latest_attempt(self, idempotency_key: str) -> Optional[ActionAttemptRecord]:
        """Retrieve the latest attempt for a given idempotency key."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT * FROM action_ledger
                        WHERE idempotency_key = ?
                        ORDER BY attempt_number DESC LIMIT 1;
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if not row:
                        return None
                    return self._row_to_record(row)
            except sqlite3.Error as exc:
                logger.error("Database error getting latest attempt for %s: %s", idempotency_key, exc)
                raise ActionLedgerError(f"Database error retrieving attempt: {exc}") from exc

    def get_attempt_by_action_id(self, action_id: str) -> Optional[ActionAttemptRecord]:
        """Retrieve the latest action attempt associated with a specific action_id."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT * FROM action_ledger
                        WHERE action_id = ?
                        ORDER BY attempt_number DESC LIMIT 1;
                        """,
                        (action_id,),
                    ).fetchone()
                    if not row:
                        return None
                    return self._row_to_record(row)
            except sqlite3.Error as exc:
                logger.error("Database error getting attempt for action_id %s: %s", action_id, exc)
                raise ActionLedgerError(f"Database error retrieving attempt: {exc}") from exc

    def update_status(
        self,
        attempt_id: str,
        new_status: ActionLedgerStatus,
        status_message: Optional[str] = None,
        error_message: Optional[str] = None,
        reconciliation_notes: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        expected_status: Optional[ActionLedgerStatus] = None,
    ) -> ActionAttemptRecord:
        """Atomically transition attempt status with strict transition validation and compare-and-swap update."""
        now_iso = datetime.now().astimezone().isoformat()
        clean_msg = sanitize_task_text(status_message)[:500] if status_message is not None else None
        clean_err = sanitize_task_text(error_message)[:1000] if error_message is not None else None
        clean_rec = sanitize_task_text(reconciliation_notes)[:1000] if reconciliation_notes is not None else None

        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM action_ledger WHERE id = ?;",
                        (attempt_id,),
                    ).fetchone()
                    if not row:
                        raise ActionLedgerNotFoundError(f"Attempt '{attempt_id}' not found.")

                    cur = self._row_to_record(row)

                    if expected_status is not None and cur.status != expected_status:
                        raise InvalidActionStateTransitionError(
                            f"Attempt '{attempt_id}' status mismatch: expected '{expected_status.value}', currently '{cur.status.value}'."
                        )

                    # Validate transition
                    allowed = VALID_LEDGER_TRANSITIONS.get(cur.status, set())
                    if new_status not in allowed:
                        raise InvalidActionStateTransitionError(
                            f"Illegal action transition from '{cur.status.value}' to '{new_status.value}' for attempt '{attempt_id}'."
                        )

                    started_at = (
                        cur.started_at
                        if cur.started_at
                        else (now_iso if new_status == ActionLedgerStatus.EXECUTING else None)
                    )
                    completed_at = (
                        now_iso if new_status in TERMINAL_LEDGER_STATUSES else cur.completed_at
                    )

                    final_meta = dict(cur.metadata)
                    if metadata_updates:
                        final_meta.update(sanitize_metadata(metadata_updates))

                    final_msg = clean_msg if clean_msg is not None else cur.status_message
                    final_err = clean_err if clean_err is not None else cur.error_message
                    final_rec = clean_rec if clean_rec is not None else cur.reconciliation_notes

                    cursor = conn.execute(
                        """
                        UPDATE action_ledger
                        SET status = ?,
                            started_at = COALESCE(started_at, ?),
                            completed_at = ?,
                            status_message = ?,
                            error_message = ?,
                            reconciliation_notes = ?,
                            metadata_json = ?
                        WHERE id = ? AND status = ?;
                        """,
                        (
                            new_status.value,
                            started_at,
                            completed_at,
                            final_msg,
                            final_err,
                            final_rec,
                            json.dumps(final_meta),
                            attempt_id,
                            cur.status.value,
                        ),
                    )

                    if cursor.rowcount == 0:
                        raise ActionLedgerConflictError(
                            f"Concurrent modification detected: Attempt '{attempt_id}' status changed from '{cur.status.value}' concurrently."
                        )

                    conn.commit()
                    logger.info("Action attempt %s transitioned: %s -> %s", attempt_id, cur.status.value, new_status.value)

                return self.get_attempt(attempt_id)  # type: ignore[return-value]

            except sqlite3.Error as exc:
                logger.error("Database error updating status for attempt %s: %s", attempt_id, exc)
                raise ActionLedgerError(f"Database error updating status: {exc}") from exc

    def reconcile_outcome(
        self,
        attempt_id: str,
        new_status: ActionLedgerStatus,
        evidence: str,
    ) -> ActionAttemptRecord:
        """Conclusively resolve an outcome_unknown action through read-only reconciliation.

        Only permitted to transition to SUCCEEDED or FAILED when verified by evidence.
        """
        if new_status not in (ActionLedgerStatus.SUCCEEDED, ActionLedgerStatus.FAILED):
            raise InvalidActionStateTransitionError(
                f"Reconciliation can only resolve to 'succeeded' or 'failed', not '{new_status.value}'."
            )

        clean_evidence = sanitize_task_text(evidence)
        return self.update_status(
            attempt_id=attempt_id,
            new_status=new_status,
            expected_status=ActionLedgerStatus.OUTCOME_UNKNOWN,
            reconciliation_notes=f"Reconciled via read-only observation: {clean_evidence}",
            status_message=f"Outcome confirmed: {new_status.value} ({clean_evidence[:100]})",
        )

    def recover_abandoned_actions(self) -> List[ActionAttemptRecord]:
        """Crash recovery for active action attempts left in non-terminal states.

        - executing -> marked outcome_unknown (never auto-re-executed).
        - awaiting_approval / approved -> marked cancelled (stale restart approvals invalidated).
        """
        now_iso = datetime.now().astimezone().isoformat()
        recovered: List[ActionAttemptRecord] = []

        with self._lock:
            try:
                with self._get_connection() as conn:
                    # 1. Recover executing actions -> outcome_unknown
                    rows_executing = conn.execute(
                        "SELECT id FROM action_ledger WHERE status = 'executing';"
                    ).fetchall()

                    for r in rows_executing:
                        att_id = r["id"]
                        msg = (
                            "Action was interrupted in 'executing' status by process shutdown. "
                            "External outcome is unconfirmed; NOT automatically re-executed."
                        )
                        conn.execute(
                            """
                            UPDATE action_ledger
                            SET status = 'outcome_unknown',
                                completed_at = ?,
                                status_message = ?,
                                reconciliation_notes = 'Preserving uncertainty: action interrupted during flight'
                            WHERE id = ? AND status = 'executing';
                            """,
                            (now_iso, msg, att_id),
                        )

                    # 2. Recover unexecuted approvals / prepared -> cancelled
                    rows_pending = conn.execute(
                        "SELECT id, status FROM action_ledger WHERE status IN ('prepared', 'awaiting_approval', 'approved');"
                    ).fetchall()

                    for r in rows_pending:
                        att_id = r["id"]
                        old_s = r["status"]
                        msg = f"Action abandoned in '{old_s}' state before execution was started; cancelled on restart."
                        conn.execute(
                            """
                            UPDATE action_ledger
                            SET status = 'cancelled',
                                completed_at = ?,
                                status_message = ?
                            WHERE id = ? AND status = ?;
                            """,
                            (now_iso, msg, att_id, old_s),
                        )

                    conn.commit()

                for r in rows_executing + rows_pending:
                    rec = self.get_attempt(r["id"])
                    if rec:
                        recovered.append(rec)

                if recovered:
                    logger.warning("Crash recovery: resolved %d abandoned action attempt(s)", len(recovered))

                return recovered

            except sqlite3.Error as exc:
                logger.error("Database error during action recovery: %s", exc)
                raise ActionLedgerError(f"Database error during action recovery: {exc}") from exc

    def list_attempts(
        self,
        task_id: Optional[str] = None,
        status: Optional[Union[ActionLedgerStatus, str]] = None,
        limit: int = 50,
    ) -> List[ActionAttemptRecord]:
        """List action attempts with optional filtering."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    query = "SELECT * FROM action_ledger WHERE 1=1"
                    params: List[Any] = []

                    if task_id:
                        query += " AND task_id = ?"
                        params.append(task_id)

                    if status:
                        s_val = status.value if isinstance(status, ActionLedgerStatus) else str(status)
                        query += " AND status = ?"
                        params.append(s_val)

                    query += " ORDER BY created_timestamp DESC LIMIT ?;"
                    params.append(max(1, limit))

                    rows = conn.execute(query, params).fetchall()
                    return [self._row_to_record(r) for r in rows]

            except sqlite3.Error as exc:
                logger.error("Database error listing action attempts: %s", exc)
                raise ActionLedgerError(f"Database error listing action attempts: {exc}") from exc


# Default global instance
default_action_ledger = ActionLedger(auto_recover=True)
