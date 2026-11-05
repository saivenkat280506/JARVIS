"""Comprehensive tests for JARVIS Persistent Task State and Crash Recovery (Milestone 09C).

Verifies:
1. Task creation and persistence across storage reinitialization.
2. Valid state transitions in the task lifecycle.
3. Invalid state transitions raise InvalidTaskStateTransitionError.
4. Terminal-state immutability (cannot modify succeeded/failed/cancelled/interrupted).
5. Explicit retry creation (linked child task without mutating completed parent).
6. Startup recovery of queued and running tasks (marked interrupted).
7. Tasks waiting for approval after restart (marked interrupted, no revival of tokens).
8. No automatic execution during recovery.
9. Concurrent state updates allow at most one successful terminal transition.
10. Database failure handling fails closed and raises TaskStorageError.
11. Sanitization of secrets and tokens from descriptions and metadata.
12. Compatibility with conversation history and approval workflows.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.runtime import AgentRuntime
from agent.tool_registry import ToolDefinition, ToolRegistry
from pydantic import BaseModel
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    PendingAction,
)
from storage.conversation import ConversationStorage
from storage.tasks import (
    InvalidTaskStateTransitionError,
    TaskNotFoundError,
    TaskRecord,
    TaskStatus,
    TaskStorage,
    TaskStorageError,
    sanitize_metadata,
    sanitize_task_text,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide an isolated temporary SQLite database path."""
    return tmp_path / "test_tasks.db"


@pytest.fixture
def task_storage(temp_db: Path) -> TaskStorage:
    """Provide an initialized TaskStorage instance without auto-recovery."""
    return TaskStorage(db_path=temp_db, auto_recover=False)


# ---------------------------------------------------------------------------
# 1. Task Creation and Persistence Across Reinitialization
# ---------------------------------------------------------------------------

def test_task_creation_and_persistence_across_reinit(temp_db: Path) -> None:
    """Ensure tasks persist durably in SQLite across manager reinitialization."""
    storage1 = TaskStorage(db_path=temp_db, auto_recover=False)
    task1 = storage1.create_task(
        task_type="web_navigation",
        description="Navigate to https://example.com/checkout",
        conversation_id="conv_test_123",
        metadata={"target": "example.com", "step": 1},
    )

    assert task1.id.startswith("task_")
    assert task1.task_type == "web_navigation"
    assert task1.status == TaskStatus.QUEUED
    assert task1.conversation_id == "conv_test_123"
    assert task1.created_at is not None
    assert task1.started_at is None
    assert task1.completed_at is None
    assert task1.metadata == {"target": "example.com", "step": 1}

    # Reinitialize new storage instance pointing to same DB
    storage2 = TaskStorage(db_path=temp_db, auto_recover=False)
    retrieved = storage2.get_task(task1.id)

    assert retrieved is not None
    assert retrieved.id == task1.id
    assert retrieved.task_type == task1.task_type
    assert retrieved.status == TaskStatus.QUEUED
    assert retrieved.description == task1.description
    assert retrieved.conversation_id == "conv_test_123"
    assert retrieved.metadata["target"] == "example.com"


# ---------------------------------------------------------------------------
# 2. Valid State Transitions
# ---------------------------------------------------------------------------

def test_valid_state_transitions(task_storage: TaskStorage) -> None:
    """Verify standard legal state lifecycle transitions."""
    # Lifecycle: queued -> running -> waiting_for_approval -> running -> succeeded
    task = task_storage.create_task(task_type="search", description="Search documentation")
    assert task.status == TaskStatus.QUEUED

    # queued -> running
    task = task_storage.update_task_status(task.id, TaskStatus.RUNNING, status_message="Started search")
    assert task.status == TaskStatus.RUNNING
    assert task.started_at is not None
    assert task.completed_at is None

    # running -> waiting_for_approval
    task = task_storage.update_task_status(
        task.id, TaskStatus.WAITING_FOR_APPROVAL, status_message="Awaiting user consent"
    )
    assert task.status == TaskStatus.WAITING_FOR_APPROVAL

    # waiting_for_approval -> running (user confirmed)
    task = task_storage.update_task_status(task.id, TaskStatus.RUNNING, status_message="User approved")
    assert task.status == TaskStatus.RUNNING

    # running -> succeeded
    task = task_storage.update_task_status(task.id, TaskStatus.SUCCEEDED, status_message="Search finished")
    assert task.status == TaskStatus.SUCCEEDED
    assert task.completed_at is not None
    assert task.is_terminal() is True

    # Test queued -> cancelled
    t_cancel = task_storage.create_task(task_type="calc", description="Calculate sum")
    t_cancel = task_storage.cancel_task(t_cancel.id, reason="User cancelled immediately")
    assert t_cancel.status == TaskStatus.CANCELLED
    assert t_cancel.is_terminal() is True

    # Test running -> failed
    t_fail = task_storage.create_task(task_type="scrape", description="Scrape web", status=TaskStatus.RUNNING)
    t_fail = task_storage.update_task_status(
        t_fail.id, TaskStatus.FAILED, error_message="Network unreachable"
    )
    assert t_fail.status == TaskStatus.FAILED
    assert t_fail.error_message == "Network unreachable"
    assert t_fail.is_terminal() is True


# ---------------------------------------------------------------------------
# 3. Invalid State Transitions
# ---------------------------------------------------------------------------

def test_invalid_state_transitions(task_storage: TaskStorage) -> None:
    """Verify illegal transitions are rejected and do not corrupt task status."""
    task = task_storage.create_task(task_type="eval", description="Evaluate prompt")
    assert task.status == TaskStatus.QUEUED

    # queued cannot jump directly to succeeded
    with pytest.raises(InvalidTaskStateTransitionError) as exc_info:
        task_storage.update_task_status(task.id, TaskStatus.SUCCEEDED)
    assert "Illegal task transition" in str(exc_info.value)

    # queued cannot jump directly to waiting_for_approval
    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.update_task_status(task.id, TaskStatus.WAITING_FOR_APPROVAL)

    # Status remains QUEUED
    refreshed = task_storage.get_task(task.id)
    assert refreshed.status == TaskStatus.QUEUED


# ---------------------------------------------------------------------------
# 4. Terminal State Immutability
# ---------------------------------------------------------------------------

def test_terminal_state_immutability(task_storage: TaskStorage) -> None:
    """Verify that tasks in terminal states cannot be transitioned to any other state."""
    # Succeeded
    t_succ = task_storage.create_task(task_type="test", description="Test task", status=TaskStatus.RUNNING)
    task_storage.update_task_status(t_succ.id, TaskStatus.SUCCEEDED)

    with pytest.raises(InvalidTaskStateTransitionError) as exc:
        task_storage.update_task_status(t_succ.id, TaskStatus.RUNNING)
    assert "terminal status" in str(exc.value)

    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.update_task_status(t_succ.id, TaskStatus.FAILED)

    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.cancel_task(t_succ.id)

    # Failed
    t_fail = task_storage.create_task(task_type="test", description="Fail task", status=TaskStatus.RUNNING)
    task_storage.update_task_status(t_fail.id, TaskStatus.FAILED, error_message="Fatal")

    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.update_task_status(t_fail.id, TaskStatus.RUNNING)

    # Cancelled
    t_canc = task_storage.create_task(task_type="test", description="Cancel task")
    task_storage.cancel_task(t_canc.id)

    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.update_task_status(t_canc.id, TaskStatus.SUCCEEDED)

    # Cannot create a new task directly in terminal status
    with pytest.raises(InvalidTaskStateTransitionError):
        task_storage.create_task(task_type="test", description="Cannot be born dead", status=TaskStatus.SUCCEEDED)


# ---------------------------------------------------------------------------
# 5. Explicit Retry Creation
# ---------------------------------------------------------------------------

def test_explicit_retry_task_creation(task_storage: TaskStorage) -> None:
    """Verify retries create new linked tasks and preserve parent immutability."""
    # Create parent task and fail it
    parent = task_storage.create_task(
        task_type="download",
        description="Download update package",
        conversation_id="conv_retry_test",
        metadata={"url": "https://example.com/pkg.tar.gz"},
        status=TaskStatus.RUNNING,
    )
    parent = task_storage.update_task_status(
        parent.id, TaskStatus.FAILED, error_message="Connection timed out"
    )
    assert parent.status == TaskStatus.FAILED

    # Create explicit retry
    retry1 = task_storage.create_retry_task(parent_task_id=parent.id)

    assert retry1.id != parent.id
    assert retry1.status == TaskStatus.QUEUED
    assert retry1.metadata["retry_of_task_id"] == parent.id
    assert retry1.metadata["retry_attempt"] == 1
    assert retry1.conversation_id == "conv_retry_test"

    # Parent remains completely untouched and failed
    parent_check = task_storage.get_task(parent.id)
    assert parent_check.status == TaskStatus.FAILED

    # Fail retry1 and retry again
    task_storage.update_task_status(retry1.id, TaskStatus.RUNNING)
    task_storage.update_task_status(retry1.id, TaskStatus.FAILED, error_message="Second failure")
    retry2 = task_storage.create_retry_task(parent_task_id=retry1.id)

    assert retry2.metadata["retry_of_task_id"] == retry1.id
    assert retry2.metadata["retry_attempt"] == 2

    # Cannot retry a succeeded task
    succ_task = task_storage.create_task(task_type="ok", description="Ok", status=TaskStatus.RUNNING)
    task_storage.update_task_status(succ_task.id, TaskStatus.SUCCEEDED)
    with pytest.raises(InvalidTaskStateTransitionError) as exc:
        task_storage.create_retry_task(parent_task_id=succ_task.id)
    assert "Only failed, cancelled, or interrupted tasks can be retried" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. Startup Recovery of Queued and Running Tasks
# ---------------------------------------------------------------------------

def test_startup_recovery_of_queued_and_running(temp_db: Path) -> None:
    """Verify abandoned queued and running tasks are marked interrupted on startup."""
    storage1 = TaskStorage(db_path=temp_db, auto_recover=False)

    t_queued = storage1.create_task(task_type="batch", description="Queued batch job", status=TaskStatus.QUEUED)
    t_running = storage1.create_task(task_type="stream", description="Streaming job", status=TaskStatus.RUNNING)
    t_done = storage1.create_task(task_type="clean", description="Finished job", status=TaskStatus.RUNNING)
    storage1.update_task_status(t_done.id, TaskStatus.SUCCEEDED)

    # Simulate process restart by instantiating new storage with auto_recover=True
    storage2 = TaskStorage(db_path=temp_db, auto_recover=True)

    rec_q = storage2.get_task(t_queued.id)
    assert rec_q.status == TaskStatus.INTERRUPTED
    assert rec_q.metadata.get("recovered_from_status") == "queued"
    assert "Execution interrupted before start" in rec_q.status_message

    rec_r = storage2.get_task(t_running.id)
    assert rec_r.status == TaskStatus.INTERRUPTED
    assert rec_r.metadata.get("recovered_from_status") == "running"
    assert "Completion status is unknown" in rec_r.status_message
    assert "external action was NOT automatically re-executed" in rec_r.status_message

    # Succeeded task is untouched
    rec_d = storage2.get_task(t_done.id)
    assert rec_d.status == TaskStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# 7. Tasks Waiting for Approval After Restart
# ---------------------------------------------------------------------------

def test_recovery_of_waiting_for_approval_tasks(temp_db: Path) -> None:
    """Verify tasks awaiting approval are marked interrupted and never auto-executed."""
    storage1 = TaskStorage(db_path=temp_db, auto_recover=False)

    t_wait = storage1.create_task(
        task_type="form_submit",
        description="Submitting sensitive form",
        status=TaskStatus.RUNNING,
    )
    storage1.update_task_status(
        t_wait.id,
        TaskStatus.WAITING_FOR_APPROVAL,
        status_message="Waiting for user confirmation",
        metadata_updates={"pending_action_id": "act_test_999"},
    )

    # Process restart with recovery
    storage2 = TaskStorage(db_path=temp_db, auto_recover=True)

    recovered = storage2.get_task(t_wait.id)
    assert recovered.status == TaskStatus.INTERRUPTED
    assert recovered.metadata.get("recovered_from_status") == "waiting_for_approval"
    assert "Pending approval tokens were invalidated on startup" in recovered.status_message
    assert "external action was NOT executed" in recovered.status_message


# ---------------------------------------------------------------------------
# 8. No Automatic Execution During Recovery
# ---------------------------------------------------------------------------

def test_no_automatic_execution_during_recovery(temp_db: Path) -> None:
    """Ensure crash recovery never triggers tool execution or external network calls."""
    storage = TaskStorage(db_path=temp_db, auto_recover=False)

    # Create several abandoned tasks
    for i in range(5):
        storage.create_task(
            task_type="external_api",
            description=f"Action {i}",
            status=TaskStatus.RUNNING,
        )

    mock_executor = MagicMock()

    with patch("storage.tasks.logger") as mock_log:
        recovered = storage.recover_interrupted_tasks()

    # Verify zero calls to mock_executor
    mock_executor.assert_not_called()
    assert len(recovered) == 5
    for task in recovered:
        assert task.status == TaskStatus.INTERRUPTED


# ---------------------------------------------------------------------------
# 9. Concurrent State Updates
# ---------------------------------------------------------------------------

def test_concurrent_state_updates(task_storage: TaskStorage) -> None:
    """Verify concurrency safety: only one thread succeeds in transitioning to terminal state."""
    task = task_storage.create_task(
        task_type="concurrency_test",
        description="Test race condition",
        status=TaskStatus.RUNNING,
    )

    successes: List[str] = []
    failures: List[str] = []
    lock = threading.Lock()

    def try_transition(target_status: TaskStatus, msg: str) -> None:
        try:
            task_storage.update_task_status(task.id, target_status, status_message=msg)
            with lock:
                successes.append(target_status.value)
        except InvalidTaskStateTransitionError as e:
            with lock:
                failures.append(str(e))

    threads = [
        threading.Thread(target=try_transition, args=(TaskStatus.SUCCEEDED, "Win Succeeded")),
        threading.Thread(target=try_transition, args=(TaskStatus.FAILED, "Win Failed")),
        threading.Thread(target=try_transition, args=(TaskStatus.CANCELLED, "Win Cancelled")),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one transition should succeed into a terminal state
    assert len(successes) == 1
    assert len(failures) == 2

    final_task = task_storage.get_task(task.id)
    assert final_task.is_terminal() is True
    assert final_task.status.value == successes[0]


# ---------------------------------------------------------------------------
# 10. Database Failure Handling
# ---------------------------------------------------------------------------

def test_database_failure_handling(tmp_path: Path) -> None:
    """Ensure database errors fail closed and raise TaskStorageError."""
    # Point storage to a path where parent cannot be created or file is corrupted
    bad_db_path = tmp_path / "non_existent_dir" / "bad.db"

    storage = TaskStorage(db_path=tmp_path / "valid.db", auto_recover=False)
    # Corrupt table intentionally
    with storage._get_connection() as conn:
        conn.execute("DROP TABLE tasks;")

    with pytest.raises(TaskStorageError):
        storage.get_task("task_nonexistent")

    with pytest.raises(TaskStorageError):
        storage.create_task("test", "test description")


# ---------------------------------------------------------------------------
# 11. Sanitization of Secrets
# ---------------------------------------------------------------------------

def test_sanitization_of_secrets(task_storage: TaskStorage, temp_db: Path) -> None:
    """Verify raw approval tokens, passwords, and bearer tokens are scrubbed from disk."""
    raw_token = "tok_a1b2c3d4e5f67890123456789abcdef0"
    raw_bearer = "bearer_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    raw_pwd = "superSecretPassword123!"

    desc = f"Action with approval_token: {raw_token} and password: {raw_pwd}"
    meta = {
        "approval_token": raw_token,
        "auth_secret": "my-secret-key",
        "nested": {
            "password": raw_pwd,
            "bearer_token": raw_bearer,
            "safe_param": "regular_value",
        },
    }

    task = task_storage.create_task(
        task_type="secure_action",
        description=desc,
        metadata=meta,
    )

    # Check via python API
    assert raw_token not in task.description
    assert "[REDACTED_TOKEN]" in task.description
    assert raw_pwd not in task.description
    assert task.metadata["approval_token"] == "[REDACTED]"
    assert task.metadata["auth_secret"] == "[REDACTED]"
    assert task.metadata["nested"]["password"] == "[REDACTED]"
    assert task.metadata["nested"]["safe_param"] == "regular_value"

    # Directly inspect raw SQLite database row
    conn = sqlite3.connect(str(temp_db))
    row = conn.execute("SELECT description, metadata_json FROM tasks WHERE id = ?", (task.id,)).fetchone()
    conn.close()

    raw_row_desc = row[0]
    raw_row_meta = row[1]

    assert raw_token not in raw_row_desc
    assert raw_pwd not in raw_row_desc
    assert raw_token not in raw_row_meta
    assert raw_pwd not in raw_row_meta
    assert "my-secret-key" not in raw_row_meta


# ---------------------------------------------------------------------------
# 12. Compatibility with Conversation History & Approval Workflows
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_runtime_task_tracking_compatibility(tmp_path: Path) -> None:
    """Verify AgentRuntime accurately tracks tasks through conversation and approval workflows."""
    db_path = tmp_path / "runtime_tasks.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    conv_storage = ConversationStorage(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)

    class DummyArgs(BaseModel):
        target: str
        action_id: Optional[str] = None
        approval_token: Optional[str] = None
        user_confirmed: Optional[bool] = False

    registry = ToolRegistry()

    # Tool requiring user confirmation
    async def sensitive_tool(target: str, **kwargs: Any) -> Dict[str, Any]:
        user_confirmed = kwargs.get("user_confirmed", False)
        approval_token = kwargs.get("approval_token")

        if not user_confirmed and not approval_token:
            action = approval_manager.create_pending_action(
                tool_name="sensitive_action",
                action_type="delete_resource",
                target=target,
                arguments={"target": target},
                description=f"Delete resource: {target}",
                metadata={"consequential_reason": "Resource deletion is permanent"},
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": action.action_id,
                "consequential_reason": "Resource deletion is permanent",
            }
        return {"success": True, "message": f"Successfully deleted {target}"}

    registry.register(
        ToolDefinition(
            name="sensitive_action",
            description="Perform a sensitive deletion requiring confirmation",
            parameters_schema=DummyArgs.model_json_schema(),
            handler=sensitive_tool,
            args_model=DummyArgs,
        )
    )

    mock_client = AsyncMock()

    runtime = AgentRuntime(
        client=mock_client,
        registry=registry,
        conversation_storage=conv_storage,
        approval_manager=approval_manager,
        task_storage=task_storage,
    )

    # Turn 1: Model calls sensitive_action -> tool requests confirmation
    mock_client.chat = AsyncMock(
        side_effect=[
            # Round 1: Model requests tool call
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "sensitive_action",
                                "arguments": {"target": "res_42"},
                            },
                        }
                    ],
                }
            },
            # Round 2: Model sees confirmation required and asks user
            {
                "message": {
                    "role": "assistant",
                    "content": "Deleting res_42 requires confirmation. Do you want to proceed?",
                }
            },
        ]
    )

    res1 = await runtime.run("Please delete res_42", conversation_id="conv_flow_1")
    assert res1.task_id is not None

    # Check task status after Turn 1: should be WAITING_FOR_APPROVAL
    task1 = task_storage.get_task(res1.task_id)
    assert task1.status == TaskStatus.WAITING_FOR_APPROVAL
    assert "Waiting for approval" in task1.status_message

    # Turn 2: User responds with affirmative confirmation "Yes, proceed"
    res2 = await runtime.run("Yes, proceed", conversation_id="conv_flow_1")

    # The waiting task should now have transitioned to SUCCEEDED
    task1_refreshed = task_storage.get_task(res1.task_id)
    assert task1_refreshed.status == TaskStatus.SUCCEEDED
    assert "Confirmed and executed" in task1_refreshed.status_message

    # Verify conversation history was preserved
    history = conv_storage.get_messages("conv_flow_1")
    assert len(history) >= 2


@pytest.mark.anyio
async def test_runtime_rejection_cancels_task(tmp_path: Path) -> None:
    """Verify that user rejecting a confirmation cancels the task."""
    db_path = tmp_path / "runtime_reject.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    conv_storage = ConversationStorage(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)

    class DummyArgs(BaseModel):
        param: str

    registry = ToolRegistry()

    async def risky_tool(param: str, **kwargs: Any) -> Dict[str, Any]:
        if not kwargs.get("approval_token"):
            action = approval_manager.create_pending_action(
                tool_name="risky_tool",
                action_type="risky_op",
                target=param,
                arguments={"param": param},
                description=f"Risky op: {param}",
                metadata={"consequential_reason": "High risk operation"},
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": action.action_id,
            }
        return {"success": True}

    registry.register(
        ToolDefinition(
            name="risky_tool",
            description="Risky tool",
            parameters_schema=DummyArgs.model_json_schema(),
            handler=risky_tool,
            args_model=DummyArgs,
        )
    )

    mock_client = AsyncMock()
    runtime = AgentRuntime(
        client=mock_client,
        registry=registry,
        conversation_storage=conv_storage,
        approval_manager=approval_manager,
        task_storage=task_storage,
    )

    # Turn 1: Request confirmation
    mock_client.chat = AsyncMock(
        side_effect=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_risk",
                            "function": {"name": "risky_tool", "arguments": {"param": "abc"}},
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Are you sure you want to run this?",
                }
            },
        ]
    )

    res1 = await runtime.run("Run risky tool", conversation_id="conv_flow_2")
    task1 = task_storage.get_task(res1.task_id)
    assert task1.status == TaskStatus.WAITING_FOR_APPROVAL

    # Turn 2: User explicitly rejects: "No, stop"
    res2 = await runtime.run("No, stop", conversation_id="conv_flow_2")

    task1_refreshed = task_storage.get_task(res1.task_id)
    assert task1_refreshed.status == TaskStatus.CANCELLED
    assert "Understood. The action has been cancelled." in res2.final_response
