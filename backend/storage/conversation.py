"""Persistent SQLite conversation storage and bounded context history for JARVIS.

Provides reliable multi-turn conversation storage in data/jarvis.db with:
- Schema for conversations and ordered messages.
- Full preservation of user, assistant, tool-call, and tool-result messages.
- Complete turn-aware context window trimming (protecting tool call/result dependencies).
- Strict redaction of approval tokens, passwords, and sensitive credentials.
- Parameterized SQL, transaction safety, and clean conversation lifecycle management.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    """Metadata record for a persistent conversation thread."""

    id: str
    title: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MessageRecord:
    """Structured representation of a message stored in SQLite."""

    conversation_id: str
    role: str
    content: str
    id: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    order_index: int = 0
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"msg_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_ollama_dict(self) -> Dict[str, Any]:
        """Convert into Ollama-compatible message dictionary format."""
        msg: Dict[str, Any] = {
            "role": self.role,
            "content": self.content or "",
        }
        if self.tool_calls:
            msg["tool_calls"] = copy.deepcopy(self.tool_calls)
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


def sanitize_conversation_text(text: str) -> str:
    """Scrub approval tokens, passwords, and secrets before persisting to conversation history."""
    if not text:
        return text
    # 1. Redact cryptographic approval tokens (tok_...)
    scrubbed = re.sub(r"\btok_[0-9a-fA-F]{16,64}\b", "[REDACTED_TOKEN]", text)
    # 2. Redact JSON approval_token values
    scrubbed = re.sub(r'("approval_token"\s*:\s*)"[^"]+"', r'\1"[REDACTED_TOKEN]"', scrubbed)
    # 3. Redact password fields in JSON or key-value pairs
    scrubbed = re.sub(r'("password"\s*:\s*)"[^"]+"', r'\1"[REDACTED_PASSWORD]"', scrubbed)
    scrubbed = re.sub(
        r'(password\s*=\s*)[^\s,)]+', r"\1[REDACTED_PASSWORD]", scrubbed, flags=re.IGNORECASE
    )
    return scrubbed


def sanitize_tool_calls(
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Scrub sensitive fields from tool call argument dictionaries before persisting."""
    if not tool_calls:
        return None
    sanitized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        tc_copy = copy.deepcopy(tc)
        fn = tc_copy.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, dict):
            for k in list(args.keys()):
                k_lower = k.lower()
                if "token" in k_lower or "secret" in k_lower:
                    args[k] = "[REDACTED_TOKEN]"
                elif "password" in k_lower:
                    args[k] = "[REDACTED_PASSWORD]"
        elif isinstance(args, str):
            fn["arguments"] = sanitize_conversation_text(args)
        sanitized.append(tc_copy)
    return sanitized


class ConversationStorage:
    """Thread-safe SQLite storage for JARVIS conversation sessions and message history."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = (db_path or settings.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create conversations and messages tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    order_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_order
                ON messages (conversation_id, order_index ASC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations (updated_at DESC);
            """)

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationRecord:
        """Create a new conversation record."""
        cid = (conversation_id or f"conv_{uuid.uuid4().hex[:12]}").strip()
        now_iso = datetime.now().astimezone().isoformat()
        conv_title = (title or "New Conversation").strip()
        meta_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?);
                """,
                (cid, conv_title, now_iso, now_iso, meta_json),
            )

        return ConversationRecord(
            id=cid,
            title=conv_title,
            created_at=now_iso,
            updated_at=now_iso,
            metadata=metadata or {},
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Retrieve conversation record with message count."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.*, COUNT(m.id) AS msg_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.id = ?
                GROUP BY c.id;
                """,
                (conversation_id.strip(),),
            )
            row = cursor.fetchone()
            if not row:
                return None

            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}

            return ConversationRecord(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=meta,
                message_count=row["msg_count"],
            )

    def get_or_create_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
    ) -> ConversationRecord:
        """Get existing conversation or create if not present."""
        existing = self.get_conversation(conversation_id)
        if existing:
            return existing
        return self.create_conversation(conversation_id=conversation_id, title=title)

    def list_conversations(self, limit: int = 50) -> List[ConversationRecord]:
        """List recent conversations sorted by latest activity."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.*, COUNT(m.id) AS msg_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results: List[ConversationRecord] = []
            for row in rows:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    meta = {}
                results.append(
                    ConversationRecord(
                        id=row["id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        metadata=meta,
                        message_count=row["msg_count"],
                    )
                )
            return results

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its associated messages."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?;",
                (conversation_id.strip(),),
            )
            return cursor.rowcount > 0

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear all messages from a conversation while preserving the conversation thread."""
        now_iso = datetime.now().astimezone().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?;",
                (conversation_id.strip(),),
            )
            cursor = conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                (now_iso, conversation_id.strip()),
            )
            return cursor.rowcount > 0

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> MessageRecord:
        """Append a single message to a conversation thread."""
        records = self.append_turn(
            conversation_id=conversation_id,
            messages=[
                {
                    "role": role,
                    "content": content,
                    "tool_calls": tool_calls,
                    "tool_call_id": tool_call_id,
                    "name": name,
                }
            ],
        )
        return records[0]

    def append_turn(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any] | MessageRecord],
        update_title: bool = True,
    ) -> List[MessageRecord]:
        """Atomically append a list of messages representing a conversational turn.

        Ensures atomic persistence, sequential ordering indices, secret scrubbing,
        and updates the conversation timestamp and title.
        """
        if not messages:
            return []

        cid = conversation_id.strip()
        self.get_or_create_conversation(cid)

        records: List[MessageRecord] = []
        now_iso = datetime.now().astimezone().isoformat()

        with self._get_connection() as conn:
            # 1. Determine next order index
            cursor = conn.execute(
                "SELECT COALESCE(MAX(order_index), -1) + 1 AS next_idx FROM messages WHERE conversation_id = ?;",
                (cid,),
            )
            start_idx = cursor.fetchone()["next_idx"]

            # 2. Insert messages
            first_user_text: Optional[str] = None
            for idx, msg in enumerate(messages, start=start_idx):
                mid = f"msg_{uuid.uuid4().hex[:12]}"
                if isinstance(msg, MessageRecord):
                    role = str(msg.role or "user").strip()
                    raw_content = str(msg.content or "")
                    raw_tc = msg.tool_calls
                    tcid = msg.tool_call_id
                    fn_name = msg.name
                else:
                    role = str(msg.get("role", "user")).strip()
                    raw_content = str(msg.get("content") or "")
                    raw_tc = msg.get("tool_calls")
                    tcid = msg.get("tool_call_id")
                    fn_name = msg.get("name")

                clean_content = sanitize_conversation_text(raw_content)
                clean_tc = sanitize_tool_calls(raw_tc)
                tc_json = json.dumps(clean_tc) if clean_tc else None

                if role == "user" and first_user_text is None and clean_content:
                    first_user_text = clean_content

                conn.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, role, content,
                        tool_calls_json, tool_call_id, name,
                        order_index, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        mid,
                        cid,
                        role,
                        clean_content,
                        tc_json,
                        tcid,
                        fn_name,
                        idx,
                        now_iso,
                    ),
                )

                records.append(
                    MessageRecord(
                        id=mid,
                        conversation_id=cid,
                        role=role,
                        content=clean_content,
                        tool_calls=clean_tc,
                        tool_call_id=tcid,
                        name=fn_name,
                        order_index=idx,
                        created_at=now_iso,
                    )
                )

            # 3. Update conversation updated_at and title if currently generic
            if first_user_text:
                title_snippet = first_user_text[:40].strip()
                conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?,
                        title = CASE WHEN title = 'New Conversation' THEN ? ELSE title END
                    WHERE id = ?;
                    """,
                    (now_iso, title_snippet, cid),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                    (now_iso, cid),
                )

        return records

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[MessageRecord]:
        """Retrieve all messages for a conversation ordered sequentially."""
        cid = conversation_id.strip()
        with self._get_connection() as conn:
            if limit is not None and limit > 0:
                cursor = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE conversation_id = ?
                        ORDER BY order_index DESC
                        LIMIT ?
                    ) ORDER BY order_index ASC;
                    """,
                    (cid, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY order_index ASC;
                    """,
                    (cid,),
                )

            rows = cursor.fetchall()
            records: List[MessageRecord] = []
            for row in rows:
                tc = None
                if row["tool_calls_json"]:
                    try:
                        tc = json.loads(row["tool_calls_json"])
                    except Exception:
                        tc = None

                records.append(
                    MessageRecord(
                        id=row["id"],
                        conversation_id=row["conversation_id"],
                        role=row["role"],
                        content=row["content"],
                        tool_calls=tc,
                        tool_call_id=row["tool_call_id"],
                        name=row["name"],
                        order_index=row["order_index"],
                        created_at=row["created_at"],
                    )
                )
            return records

    def get_bounded_history(
        self,
        conversation_id: str,
        max_turns: int = 5,
        max_chars: int = 6000,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversational turns bounded by turns and characters.

        Safety properties:
        - Bounded by complete conversational turns starting at a 'user' message.
        - Preserves assistant tool-calls and corresponding tool responses together.
        - Trims oldest complete turns when character or turn budget is exceeded.
        - Returns list of message dictionaries ready for Ollama chat payload.
        """
        all_msgs = self.get_messages(conversation_id)
        if not all_msgs:
            return []

        # 1. Group messages into atomic conversational turns
        # A turn begins with a 'user' message and includes all subsequent assistant/tool messages
        turns: List[List[MessageRecord]] = []
        current_turn: List[MessageRecord] = []

        for m in all_msgs:
            if m.role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [m]
            else:
                current_turn.append(m)

        if current_turn:
            turns.append(current_turn)

        if not turns:
            return []

        # 2. Select the most recent K turns
        budget_turns = max(1, max_turns)
        selected_turns = turns[-budget_turns:]

        # 3. Apply character budget constraint (dropping oldest complete turns if needed)
        def calc_chars(turn_list: List[List[MessageRecord]]) -> int:
            return sum(
                len(m.content or "") + (len(json.dumps(m.tool_calls)) if m.tool_calls else 0)
                for t in turn_list
                for m in t
            )

        while len(selected_turns) > 1 and calc_chars(selected_turns) > max_chars:
            selected_turns.pop(0)

        # 4. Flatten selected turns into chronological message records
        flattened: List[MessageRecord] = [m for t in selected_turns for m in t]

        # 5. Convert to Ollama dictionary format
        return [m.to_ollama_dict() for m in flattened]


# Global shared instance using project-wide database
default_conversation_storage = ConversationStorage()
