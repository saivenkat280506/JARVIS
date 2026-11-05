"""Secure Action Approval and Authorization Boundary for JARVIS.

Ensures that consequential actions (e.g. form submissions, financial transactions,
deletions, account alterations) cannot be independently authorized by model-generated
arguments. Enforces durable SQLite storage, cryptographic single-use approval tokens,
exact argument binding, strict expiration, crash recovery invalidation, and concurrency safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import hmac
import json
import logging
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from config.settings import settings

logger = logging.getLogger("jarvis.security.approval")


class SecurityAuditStorageError(Exception):
    """Raised when writing to the security audit log fails."""


class SecurityApprovalError(Exception):
    """Raised when an approval operation violates security policy or fails storage."""


# Words indicating clear affirmative confirmation
AFFIRM_WORDS: set[str] = {
    "yes",
    "confirm",
    "confirmed",
    "i confirm",
    "approve",
    "approved",
    "i approve",
    "proceed",
    "go ahead",
    "do it",
    "authorize",
    "authorized",
    "sure",
    "yep",
    "affirmative",
    "ok proceed",
    "okay proceed",
    "yes please",
}

# Words indicating clear explicit rejection
REJECT_WORDS: set[str] = {
    "no",
    "cancel",
    "stop",
    "abort",
    "don't",
    "dont",
    "do not",
    "deny",
    "denied",
    "reject",
    "rejected",
    "never mind",
    "nevermind",
    "no thanks",
}


class ApprovalStatus(str, Enum):
    """Lifecycle status of a pending consequential action."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    INVALIDATED_RESTART = "invalidated_restart"


@dataclass
class PendingAction:
    """Server-side record of a consequential action awaiting trusted user approval."""

    action_id: str
    tool_name: str
    action_type: str
    target: str
    arguments: Dict[str, Any]
    arguments_hash: str
    description: str
    created_at: float
    expires_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    approval_token: Optional[str] = None
    approval_token_hash: Optional[str] = None
    approved_channel: Optional[str] = None
    consumed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if action has passed its expiration deadline."""
        current_time = now if now is not None else time.time()
        return current_time > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "action_type": self.action_type,
            "target": self.target,
            "arguments": self.arguments,
            "arguments_hash": self.arguments_hash,
            "description": self.description,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value if isinstance(self.status, ApprovalStatus) else str(self.status),
            "approval_token": self.approval_token,
            "approved_channel": self.approved_channel,
            "consumed_at": self.consumed_at,
            "metadata": self.metadata,
        }


@dataclass
class AuditRecord:
    """Security audit log entry for authorization decisions without sensitive data."""

    timestamp: float
    event_type: str
    action_id: str
    tool_name: str
    action_type: str
    target_summary: str
    status: str
    details: str
    id: Optional[int] = None
    timestamp_iso: Optional[str] = None


def sanitize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Redact passwords, tokens, secrets from argument dictionaries."""
    sanitized: Dict[str, Any] = {}
    for k, v in arguments.items():
        k_lower = str(k).lower()
        if any(s in k_lower for s in ["pass", "secret", "token", "key", "auth", "credential"]):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_arguments(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_arguments(item) if isinstance(item, dict) else item
                for item in v
            ]
        elif isinstance(v, str):
            sanitized[k] = sanitize_audit_text(v)
        else:
            sanitized[k] = v
    return sanitized


def sanitize_audit_text(text: str) -> str:
    """Scrub raw approval tokens, passwords, and sensitive keys from audit log text."""
    if not text:
        return ""
    scrubbed = re.sub(r"\btok_[0-9a-fA-F]{16,64}\b", "[REDACTED_TOKEN]", text)
    scrubbed = re.sub(r"\bbearer[_\-\w]+\b", "[REDACTED_BEARER]", scrubbed, flags=re.IGNORECASE)
    pattern = r"((?:password|passwd|pwd|secret|token|bearer|key|credential|auth)\w*(?:[^\w\s]*\s*(?:field|input|value|box))?\s*(?:[:=]|is|with|to|\s)\s*)([^\s,;)\"'\`]+)"
    scrubbed = re.sub(pattern, r"\1[REDACTED]", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'("(?:password|secret|auth_token|token|key|credential)"\s*:\s*)"[^"]+"', r'\1"[REDACTED]"', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'("approval_token"\s*:\s*)"[^"]+"', r'\1"[REDACTED_TOKEN]"', scrubbed, flags=re.IGNORECASE)
    return scrubbed


def compute_arguments_hash(arguments: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of canonical arguments excluding authorization tokens."""
    canonical = {
        k: v
        for k, v in arguments.items()
        if k not in {"approval_token", "action_id", "user_confirmed"} and v is not None
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def classify_confirmation_intent(text: str) -> Tuple[Literal["affirm", "reject", "ambiguous"], float]:
    """Classify user utterance into unambiguous affirmation, explicit rejection, or ambiguous.

    Returns:
        Tuple[str, float]: (intent, confidence)
    """
    clean = text.strip().lower()
    clean = re.sub(r"[^\w\s]", "", clean).strip()

    if not clean:
        return "ambiguous", 0.0

    if clean in AFFIRM_WORDS:
        return "affirm", 1.0

    if clean in REJECT_WORDS:
        return "reject", 1.0

    words = clean.split()
    if len(words) <= 4:
        if any(w in {"confirm", "approve", "proceed", "authorize"} for w in words):
            if not any(neg in words for neg in {"not", "dont", "no", "never"}):
                return "affirm", 0.95
        if any(w in {"cancel", "abort", "deny", "stop"} for w in words):
            return "reject", 0.95

    return "ambiguous", 0.0


class ActionApprovalManager:
    """Durable, thread-safe, restart-safe manager enforcing trusted approval and binding for consequential actions."""

    def __init__(
        self,
        default_ttl_sec: float = 60.0,
        db_path: Optional[Path] = None,
    ) -> None:
        self.default_ttl_sec = default_ttl_sec
        self.db_path = (db_path or settings.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._token_cache: Dict[str, str] = {}  # In-memory mapping action_id -> raw token for active process
        self._init_db()
        self._recover_and_invalidate_stale_approvals()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create pending_actions and security_audit_log tables and indexes in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pending_actions (
                        action_id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        target TEXT NOT NULL,
                        arguments_json TEXT NOT NULL,
                        arguments_hash TEXT NOT NULL,
                        description TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        status TEXT NOT NULL,
                        approval_token_hash TEXT,
                        approved_channel TEXT,
                        consumed_at REAL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pending_actions_status
                    ON pending_actions (status, expires_at);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pending_actions_tool
                    ON pending_actions (tool_name, status);
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS security_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        timestamp_iso TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        action_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        target_summary TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details TEXT NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON security_audit_log (timestamp DESC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_action_id
                    ON security_audit_log (action_id);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_event_type
                    ON security_audit_log (event_type);
                """)
        except sqlite3.Error as exc:
            logger.error("Failed to initialize security approval tables in %s: %s", self.db_path, exc)
            raise SecurityAuditStorageError(f"Database error initializing security approval tables: {exc}") from exc

    def _recover_and_invalidate_stale_approvals(self) -> int:
        """Invalidate all pending or approved actions from previous process runs upon startup recovery."""
        now = time.time()
        now_iso = datetime.now().astimezone().isoformat()
        invalidated_count = 0
        try:
            with self._lock:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        """
                        SELECT action_id, tool_name, action_type, target, description, status
                        FROM pending_actions
                        WHERE status IN ('pending', 'approved');
                        """
                    ).fetchall()

                    for r in rows:
                        action_id = r["action_id"]
                        tool_name = r["tool_name"]
                        action_type = r["action_type"]
                        target = r["target"]
                        prev_status = r["status"]
                        description = r["description"]

                        conn.execute(
                            """
                            UPDATE pending_actions
                            SET status = 'invalidated_restart',
                                approval_token_hash = NULL
                            WHERE action_id = ?;
                            """,
                            (action_id,),
                        )

                        details = (
                            f"Action '{description}' in status '{prev_status}' was invalidated upon process startup recovery"
                        )
                        conn.execute(
                            """
                            INSERT INTO security_audit_log (
                                timestamp, timestamp_iso, event_type, action_id,
                                tool_name, action_type, target_summary, status, details
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            (
                                now,
                                now_iso,
                                "STARTUP_RECOVERY_INVALIDATED",
                                action_id,
                                tool_name,
                                action_type,
                                target[:100],
                                "invalidated_restart",
                                details,
                            ),
                        )
                        invalidated_count += 1

                    if invalidated_count > 0:
                        logger.info("Startup recovery: invalidated %d stale pending/approved action(s)", invalidated_count)
            return invalidated_count
        except sqlite3.Error as exc:
            logger.error("Failed to execute startup recovery invalidation: %s", exc)
            raise SecurityAuditStorageError(f"Database error during startup recovery invalidation: {exc}") from exc

    def _row_to_pending_action(self, row: sqlite3.Row) -> PendingAction:
        args_dict = json.loads(row["arguments_json"]) if row["arguments_json"] else {}
        meta_dict = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        status_val = (
            ApprovalStatus(row["status"])
            if row["status"] in ApprovalStatus._value2member_map_
            else ApprovalStatus.REJECTED
        )

        return PendingAction(
            action_id=row["action_id"],
            tool_name=row["tool_name"],
            action_type=row["action_type"],
            target=row["target"],
            arguments=args_dict,
            arguments_hash=row["arguments_hash"],
            description=row["description"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            status=status_val,
            approval_token=self._token_cache.get(row["action_id"]),
            approval_token_hash=row["approval_token_hash"],
            approved_channel=row["approved_channel"],
            consumed_at=row["consumed_at"],
            metadata=meta_dict,
        )

    def _record_audit(
        self,
        event_type: str,
        action: PendingAction,
        details: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> AuditRecord:
        """Append a structured audit entry without sensitive credentials to SQLite."""
        now = time.time()
        now_iso = datetime.now().astimezone().isoformat()
        clean_target = action.target[:100]
        clean_details = sanitize_audit_text(details)[:500]
        status_val = action.status.value if isinstance(action.status, ApprovalStatus) else str(action.status)

        record = AuditRecord(
            timestamp=now,
            event_type=event_type,
            action_id=action.action_id,
            tool_name=action.tool_name,
            action_type=action.action_type,
            target_summary=clean_target,
            status=status_val,
            details=clean_details,
            timestamp_iso=now_iso,
        )

        def _do_insert(c: sqlite3.Connection):
            cursor = c.execute(
                """
                INSERT INTO security_audit_log (
                    timestamp, timestamp_iso, event_type, action_id,
                    tool_name, action_type, target_summary, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    now,
                    now_iso,
                    event_type,
                    action.action_id,
                    action.tool_name,
                    action.action_type,
                    clean_target,
                    status_val,
                    clean_details,
                ),
            )
            record.id = cursor.lastrowid

        try:
            if conn is not None:
                _do_insert(conn)
            else:
                with self._get_connection() as c:
                    _do_insert(c)
        except sqlite3.Error as exc:
            logger.error("Database error persisting audit record (%s): %s", event_type, exc)
            raise SecurityAuditStorageError(f"Failed to record security audit log: {exc}") from exc

        return record

    def _log_unauthorized_attempt(
        self,
        action_id: str,
        tool_name: str,
        details: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> AuditRecord:
        now = time.time()
        now_iso = datetime.now().astimezone().isoformat()
        clean_details = sanitize_audit_text(details)[:500]

        record = AuditRecord(
            timestamp=now,
            event_type="UNAUTHORIZED_ATTEMPT",
            action_id=action_id,
            tool_name=tool_name,
            action_type="unknown",
            target_summary="unknown",
            status="rejected",
            details=clean_details,
            timestamp_iso=now_iso,
        )

        def _do_insert(c: sqlite3.Connection):
            cursor = c.execute(
                """
                INSERT INTO security_audit_log (
                    timestamp, timestamp_iso, event_type, action_id,
                    tool_name, action_type, target_summary, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    now,
                    now_iso,
                    "UNAUTHORIZED_ATTEMPT",
                    action_id,
                    tool_name,
                    "unknown",
                    "unknown",
                    "rejected",
                    clean_details,
                ),
            )
            record.id = cursor.lastrowid

        try:
            if conn is not None:
                _do_insert(conn)
            else:
                with self._get_connection() as c:
                    _do_insert(c)
        except sqlite3.Error as exc:
            logger.error("Database error persisting unauthorized attempt audit: %s", exc)
            raise SecurityAuditStorageError(f"Failed to record security audit log: {exc}") from exc

        return record

    def create_pending_action(
        self,
        tool_name: str,
        action_type: str,
        target: str,
        arguments: Dict[str, Any],
        description: str,
        ttl_sec: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PendingAction:
        """Register a new server-side pending consequential action with persistent SQLite records."""
        now = time.time()
        ttl = ttl_sec if ttl_sec is not None else self.default_ttl_sec
        action_id = f"act_{secrets.token_hex(8)}"
        args_hash = compute_arguments_hash(arguments)
        sanitized_args = sanitize_arguments(arguments)
        sanitized_args_json = json.dumps(sanitized_args, sort_keys=True)
        sanitized_description = sanitize_audit_text(description)
        sanitized_target = sanitize_audit_text(target)[:150]
        meta_json = json.dumps(metadata or {})

        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Check existing pending actions for the same tool
                    rows = conn.execute(
                        """
                        SELECT * FROM pending_actions
                        WHERE tool_name = ? AND status = 'pending';
                        """,
                        (tool_name,),
                    ).fetchall()

                    for r in rows:
                        existing = self._row_to_pending_action(r)
                        if not existing.is_expired(now) and existing.arguments_hash == args_hash:
                            # Same action already pending and valid; reuse existing record
                            return existing
                        else:
                            # Invalidate existing action because parameters changed or expired
                            conn.execute(
                                """
                                UPDATE pending_actions
                                SET status = 'rejected'
                                WHERE action_id = ?;
                                """,
                                (existing.action_id,),
                            )
                            existing.status = ApprovalStatus.REJECTED
                            self._record_audit(
                                event_type="INVALIDATED_ACTION_CHANGED",
                                action=existing,
                                details="Invalidated because action changed or expired before approval",
                                conn=conn,
                            )

                    action = PendingAction(
                        action_id=action_id,
                        tool_name=tool_name,
                        action_type=action_type,
                        target=sanitized_target,
                        arguments=sanitized_args,
                        arguments_hash=args_hash,
                        description=sanitized_description,
                        created_at=now,
                        expires_at=now + ttl,
                        status=ApprovalStatus.PENDING,
                        metadata=metadata or {},
                    )

                    conn.execute(
                        """
                        INSERT INTO pending_actions (
                            action_id, tool_name, action_type, target,
                            arguments_json, arguments_hash, description,
                            created_at, expires_at, status, approval_token_hash,
                            approved_channel, consumed_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?);
                        """,
                        (
                            action.action_id,
                            action.tool_name,
                            action.action_type,
                            action.target,
                            sanitized_args_json,
                            action.arguments_hash,
                            action.description,
                            action.created_at,
                            action.expires_at,
                            action.status.value,
                            meta_json,
                        ),
                    )

                    self._record_audit(
                        event_type="ACTION_REQUESTED",
                        action=action,
                        details=f"Pending approval requested for: {sanitized_description}",
                        conn=conn,
                    )

                    logger.info("Created pending action %s for tool '%s'", action_id, tool_name)
                    return action

            except sqlite3.Error as exc:
                logger.error("Database error creating pending action: %s", exc)
                raise SecurityAuditStorageError(f"Database error creating pending action: {exc}") from exc

    def get_pending_action(self, action_id: str) -> Optional[PendingAction]:
        """Retrieve a pending action record by ID from persistent storage."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM pending_actions WHERE action_id = ?;",
                        (action_id,),
                    ).fetchone()
                    if not row:
                        return None

                    action = self._row_to_pending_action(row)
                    if action_id in self._token_cache and action.status == ApprovalStatus.APPROVED:
                        action.approval_token = self._token_cache[action_id]

                    if action.is_expired():
                        if action.status == ApprovalStatus.PENDING:
                            action.status = ApprovalStatus.EXPIRED
                            conn.execute(
                                "UPDATE pending_actions SET status = 'expired' WHERE action_id = ?;",
                                (action_id,),
                            )
                            self._record_audit(
                                event_type="ACTION_EXPIRED",
                                action=action,
                                details="Action expired before approval or execution",
                                conn=conn,
                            )
                    return action
            except sqlite3.Error as exc:
                logger.error("Database error retrieving pending action %s: %s", action_id, exc)
                raise SecurityAuditStorageError(f"Database error retrieving pending action: {exc}") from exc

    def get_latest_pending(self, tool_name: Optional[str] = None) -> Optional[PendingAction]:
        """Retrieve the most recent active pending action awaiting approval."""
        now = time.time()
        with self._lock:
            try:
                with self._get_connection() as conn:
                    if tool_name:
                        rows = conn.execute(
                            """
                            SELECT * FROM pending_actions
                            WHERE tool_name = ? AND status = 'pending'
                            ORDER BY created_at DESC;
                            """,
                            (tool_name,),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT * FROM pending_actions
                            WHERE status = 'pending'
                            ORDER BY created_at DESC;
                            """
                        ).fetchall()

                    for r in rows:
                        action = self._row_to_pending_action(r)
                        if action.is_expired(now):
                            action.status = ApprovalStatus.EXPIRED
                            conn.execute(
                                "UPDATE pending_actions SET status = 'expired' WHERE action_id = ?;",
                                (action.action_id,),
                            )
                            self._record_audit(
                                event_type="ACTION_EXPIRED",
                                action=action,
                                details="Action expired before approval",
                                conn=conn,
                            )
                            continue
                        return action
                    return None
            except sqlite3.Error as exc:
                logger.error("Database error retrieving latest pending action: %s", exc)
                raise SecurityAuditStorageError(f"Database error retrieving latest pending action: {exc}") from exc

    def get_latest_action(self, tool_name: Optional[str] = None) -> Optional[PendingAction]:
        """Retrieve the most recent action record regardless of status (for deduplication / audit inspection)."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    if tool_name:
                        row = conn.execute(
                            """
                            SELECT * FROM pending_actions
                            WHERE tool_name = ?
                            ORDER BY created_at DESC LIMIT 1;
                            """,
                            (tool_name,),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            """
                            SELECT * FROM pending_actions
                            ORDER BY created_at DESC LIMIT 1;
                            """
                        ).fetchone()
                    if not row:
                        return None
                    return self._row_to_pending_action(row)
            except sqlite3.Error as exc:
                logger.error("Database error retrieving latest action: %s", exc)
                raise SecurityAuditStorageError(f"Database error retrieving latest action: {exc}") from exc

    def approve_action(
        self,
        action_id: str,
        channel: str = "trusted_ui",
    ) -> Tuple[bool, str, Optional[str]]:
        """Grant explicit user approval to a pending action and generate a single-use token."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM pending_actions WHERE action_id = ?;",
                        (action_id,),
                    ).fetchone()
                    if not row:
                        return False, f"Action ID '{action_id}' not found.", None

                    action = self._row_to_pending_action(row)
                    if action.is_expired():
                        action.status = ApprovalStatus.EXPIRED
                        conn.execute(
                            "UPDATE pending_actions SET status = 'expired' WHERE action_id = ?;",
                            (action_id,),
                        )
                        self._record_audit(
                            event_type="APPROVE_FAILED_EXPIRED",
                            action=action,
                            details=f"Cannot approve expired action (channel: {channel})",
                            conn=conn,
                        )
                        return False, "Action has expired and cannot be approved.", None

                    if action.status != ApprovalStatus.PENDING:
                        self._record_audit(
                            event_type="APPROVE_FAILED_INVALID_STATE",
                            action=action,
                            details=f"Cannot approve action in state '{action.status.value}'",
                            conn=conn,
                        )
                        return (
                            False,
                            f"Action is not in pending state (currently: {action.status.value}).",
                            None,
                        )

                    token = f"tok_{secrets.token_hex(16)}"
                    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

                    cursor = conn.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'approved',
                            approval_token_hash = ?,
                            approved_channel = ?
                        WHERE action_id = ? AND status = 'pending';
                        """,
                        (token_hash, channel, action_id),
                    )
                    if cursor.rowcount == 0:
                        return False, "Action state changed concurrently before approval.", None

                    self._token_cache[action_id] = token
                    action.status = ApprovalStatus.APPROVED
                    action.approval_token = token
                    action.approval_token_hash = token_hash
                    action.approved_channel = channel

                    self._record_audit(
                        event_type="ACTION_APPROVED",
                        action=action,
                        details=f"Approved by user via channel '{channel}'",
                        conn=conn,
                    )
                    logger.info("Action %s approved via %s", action_id, channel)
                    return True, "Action successfully approved.", token

            except sqlite3.Error as exc:
                logger.error("Database error approving action %s: %s", action_id, exc)
                raise SecurityAuditStorageError(f"Database error approving action: {exc}") from exc

    def reject_action(
        self,
        action_id: str,
        channel: str = "trusted_ui",
        reason: str = "User denied confirmation",
    ) -> bool:
        """Explicitly reject a pending action."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM pending_actions WHERE action_id = ?;",
                        (action_id,),
                    ).fetchone()
                    if not row:
                        return False

                    action = self._row_to_pending_action(row)
                    conn.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'rejected',
                            approval_token_hash = NULL
                        WHERE action_id = ?;
                        """,
                        (action_id,),
                    )
                    self._token_cache.pop(action_id, None)
                    action.status = ApprovalStatus.REJECTED

                    self._record_audit(
                        event_type="ACTION_REJECTED",
                        action=action,
                        details=f"Rejected via '{channel}': {reason}",
                        conn=conn,
                    )
                    logger.info("Action %s rejected via %s: %s", action_id, channel, reason)
                    return True

            except sqlite3.Error as exc:
                logger.error("Database error rejecting action %s: %s", action_id, exc)
                raise SecurityAuditStorageError(f"Database error rejecting action: {exc}") from exc

    def verify_and_consume_approval(
        self,
        action_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        approval_token: str,
    ) -> Tuple[bool, str]:
        """Strictly verify approval token, exact argument binding, single-use, and atomically consume token."""
        if not action_id or not approval_token:
            return False, "Missing action_id or approval_token for consequential action."

        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM pending_actions WHERE action_id = ?;",
                        (action_id,),
                    ).fetchone()
                    if not row:
                        self._log_unauthorized_attempt(action_id, tool_name, "Non-existent action ID", conn=conn)
                        return False, f"Action ID '{action_id}' does not exist."

                    action = self._row_to_pending_action(row)

                    # 1. Expiration check
                    if action.is_expired():
                        action.status = ApprovalStatus.EXPIRED
                        conn.execute(
                            "UPDATE pending_actions SET status = 'expired' WHERE action_id = ?;",
                            (action_id,),
                        )
                        self._record_audit(
                            event_type="CONSUME_FAILED_EXPIRED",
                            action=action,
                            details="Attempted to consume expired approval",
                            conn=conn,
                        )
                        return False, "Approval has expired. Action must be re-requested."

                    # 2. Replay check (already consumed)
                    if action.status == ApprovalStatus.CONSUMED:
                        self._record_audit(
                            event_type="REPLAY_ATTACK_BLOCKED",
                            action=action,
                            details="Attempted reuse of already consumed approval token",
                            conn=conn,
                        )
                        return False, "Approval token has already been consumed (replay attack rejected)."

                    # 3. Approval state check
                    if action.status != ApprovalStatus.APPROVED:
                        self._record_audit(
                            event_type="UNAPPROVED_EXECUTION_BLOCKED",
                            action=action,
                            details=f"Attempted execution in unapproved state '{action.status.value}'",
                            conn=conn,
                        )
                        return False, f"Action has not been approved by user (state: {action.status.value})."

                    # 4. Tool binding check
                    if action.tool_name != tool_name:
                        self._record_audit(
                            event_type="TOOL_MISMATCH_BLOCKED",
                            action=action,
                            details=f"Token bound to tool '{action.tool_name}', attempted on '{tool_name}'",
                            conn=conn,
                        )
                        return False, f"Approval is not valid for tool '{tool_name}'."

                    # 5. Argument binding check (tamper resistance)
                    incoming_hash = compute_arguments_hash(arguments)
                    if incoming_hash != action.arguments_hash:
                        action.status = ApprovalStatus.REJECTED
                        conn.execute(
                            """
                            UPDATE pending_actions
                            SET status = 'rejected', approval_token_hash = NULL
                            WHERE action_id = ?;
                            """,
                            (action_id,),
                        )
                        self._token_cache.pop(action_id, None)
                        self._record_audit(
                            event_type="ARGUMENT_TAMPERING_BLOCKED",
                            action=action,
                            details="Action arguments do not match the approved parameters; invalidated",
                            conn=conn,
                        )
                        return (
                            False,
                            "Action arguments have been altered from the approved parameters (tampering rejected; approval invalidated).",
                        )

                    # 6. Constant-time token match
                    token_hash = hashlib.sha256(approval_token.encode("utf-8")).hexdigest()
                    if not action.approval_token_hash or not hmac.compare_digest(token_hash, action.approval_token_hash):
                        self._record_audit(
                            event_type="INVALID_TOKEN_BLOCKED",
                            action=action,
                            details="Approval token does not match server record",
                            conn=conn,
                        )
                        return False, "Invalid or fabricated approval token."

                    # 7. Atomic state transition in SQLite: APPROVED -> CONSUMED
                    now = time.time()
                    cursor = conn.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'consumed',
                            consumed_at = ?,
                            approval_token_hash = NULL
                        WHERE action_id = ? AND status = 'approved';
                        """,
                        (now, action_id),
                    )
                    if cursor.rowcount == 0:
                        # Another thread or process consumed it concurrently
                        self._record_audit(
                            event_type="REPLAY_ATTACK_BLOCKED",
                            action=action,
                            details="Approval token consumed concurrently",
                            conn=conn,
                        )
                        return False, "Approval token has already been consumed (replay attack rejected)."

                    self._token_cache.pop(action_id, None)
                    action.status = ApprovalStatus.CONSUMED
                    action.consumed_at = now

                    self._record_audit(
                        event_type="ACTION_CONSUMED",
                        action=action,
                        details="Approval successfully verified and consumed for execution",
                        conn=conn,
                    )
                    logger.info("Approval token consumed successfully for action %s", action_id)
                    return True, "Approval verified and consumed."

            except sqlite3.Error as exc:
                logger.error("Database error verifying/consuming approval %s: %s", action_id, exc)
                return False, f"Security audit storage failure: {exc}"

    def record_execution_outcome(
        self,
        action_id: str,
        tool_name: Optional[Any] = None,
        success: bool = True,
        details: str = "",
        error_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Record final execution outcome of a consequential action in security audit trail."""
        # Handle positional call without tool_name: record_execution_outcome(action_id, success, details)
        if isinstance(tool_name, bool):
            details = str(success) if not details else details
            success = tool_name
            tool_name = None

        if error_message and not details:
            details = error_message
        if not details:
            details = "Action completed" if success else "Action failed"

        event_type = "EXECUTION_SUCCESS" if success else "EXECUTION_FAILED"
        with self._lock:
            try:
                action = self.get_pending_action(action_id)
                if action:
                    self._record_audit(
                        event_type=event_type,
                        action=action,
                        details=details,
                    )
                else:
                    self._log_unauthorized_attempt(
                        action_id=action_id,
                        tool_name=str(tool_name or "unknown"),
                        details=f"Execution outcome ({event_type}): {details}",
                    )
            except Exception as exc:
                logger.error("Failed to record execution outcome for %s: %s", action_id, exc)
                raise SecurityAuditStorageError(f"Database error recording execution outcome: {exc}") from exc

    def get_audit_log(self, limit: Optional[int] = None) -> List[AuditRecord]:
        """Return full audit log from SQLite database."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    query = (
                        "SELECT timestamp, event_type, action_id, tool_name, "
                        "action_type, target_summary, status, details, id, timestamp_iso "
                        "FROM security_audit_log ORDER BY id ASC"
                    )
                    if limit is not None:
                        query += f" LIMIT {int(limit)}"
                    rows = conn.execute(query).fetchall()
                    return [
                        AuditRecord(
                            timestamp=r["timestamp"],
                            event_type=r["event_type"],
                            action_id=r["action_id"],
                            tool_name=r["tool_name"],
                            action_type=r["action_type"],
                            target_summary=r["target_summary"],
                            status=r["status"],
                            details=r["details"],
                            id=r["id"],
                            timestamp_iso=r["timestamp_iso"],
                        )
                        for r in rows
                    ]
            except sqlite3.Error as exc:
                logger.error("Database error retrieving audit log: %s", exc)
                raise SecurityAuditStorageError(f"Database error retrieving audit log: {exc}") from exc


# Global singleton instance
default_approval_manager = ActionApprovalManager()
