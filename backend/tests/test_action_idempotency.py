"""Comprehensive tests for JARVIS Consequential Action Idempotency and Outcome Reconciliation (Milestone 09D).

Verifies:
1. Duplicate requests for the same logical action are deduplicated.
2. Distinct intentional actions with identical arguments proceed independently.
3. Concurrent attempts using the same idempotency key reject duplicate executions.
4. Crash after marking an action executing recovers to outcome_unknown without re-execution.
5. Crash after simulated external effect recovers to outcome_unknown and reconciles via read-only observation.
6. Timeout preserves unknown outcome and rejects assumptions of failure.
7. Read-only reconciliation conclusively resolves outcomes when evidence exists.
8. Approval authorization remains strictly separated from idempotency (replay, tamper, expiration blocked).
9. Restart invalidation of stale approvals transitions pending actions to cancelled.
10. No blind replay of uncertain consequential actions without explicit decision.
11. Database failures fail closed and maintain transaction integrity.
12. End-to-end runtime integration deduplicates repetitive execution requests.
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
from pydantic import BaseModel

from agent.runtime import AgentRuntime
from agent.tool_registry import ToolDefinition, ToolRegistry
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    PendingAction,
)
from storage.action_ledger import (
    ActionAttemptRecord,
    ActionLedger,
    ActionLedgerConflictError,
    ActionLedgerError,
    ActionLedgerNotFoundError,
    ActionLedgerStatus,
    ActionUncertaintyError,
    InvalidActionStateTransitionError,
    RetrySafety,
    generate_idempotency_key,
    normalize_and_hash_arguments,
)
from storage.conversation import ConversationStorage
from storage.tasks import TaskStorage
from tools.browser_automation import BrowserAutomationController
from tools.cdp_client import CDPClient


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Isolated temporary SQLite database."""
    return tmp_path / "test_idempotency.db"


@pytest.fixture
def ledger(temp_db: Path) -> ActionLedger:
    """Initialized ActionLedger instance without auto-recovery."""
    return ActionLedger(db_path=temp_db, auto_recover=False)


# ---------------------------------------------------------------------------
# 1. Duplicate Requests for the Same Logical Action
# ---------------------------------------------------------------------------

def test_duplicate_requests_same_logical_action(ledger: ActionLedger) -> None:
    """Verify that repeating the same logical action returns cached success and skips re-execution."""
    task_id = "task_order_1"
    step_name = "submit_checkout"
    tool_name = "browser_interact"
    args = {"action": "click", "selector": "#btn-submit-order"}

    # Attempt 1: Prepared -> Executing -> Succeeded
    attempt1, is_cached = ledger.prepare_action(
        task_id=task_id,
        step_name=step_name,
        tool_name=tool_name,
        arguments=args,
        retry_safety=RetrySafety.NON_IDEMPOTENT,
    )
    assert is_cached is False
    assert attempt1.attempt_number == 1
    assert attempt1.status == ActionLedgerStatus.PREPARED

    ledger.update_status(attempt1.id, ActionLedgerStatus.EXECUTING)
    ledger.update_status(
        attempt1.id,
        ActionLedgerStatus.SUCCEEDED,
        status_message="Order #1042 successfully placed",
    )

    # Attempt 2: Same logical action with identical task_id and step_name
    attempt2, is_cached2 = ledger.prepare_action(
        task_id=task_id,
        step_name=step_name,
        tool_name=tool_name,
        arguments=args,
        retry_safety=RetrySafety.NON_IDEMPOTENT,
    )

    # Must detect existing success and return cached attempt without creating new attempt
    assert is_cached2 is True
    assert attempt2.id == attempt1.id
    assert attempt2.attempt_number == 1
    assert attempt2.status == ActionLedgerStatus.SUCCEEDED
    assert "Order #1042" in attempt2.status_message


# ---------------------------------------------------------------------------
# 2. Distinct Intentional Actions with Identical Arguments
# ---------------------------------------------------------------------------

def test_distinct_intentional_actions_with_identical_arguments(ledger: ActionLedger) -> None:
    """Verify that two intentional separate actions with identical arguments are not conflated."""
    args = {"action": "click", "selector": "#add-item-to-cart"}

    # Action 1: Add item 1
    attempt1, cached1 = ledger.prepare_action(
        task_id="task_cart_1",
        step_name="add_first_coffee",
        tool_name="browser_interact",
        arguments=args,
    )
    assert cached1 is False
    ledger.update_status(attempt1.id, ActionLedgerStatus.EXECUTING)
    ledger.update_status(attempt1.id, ActionLedgerStatus.SUCCEEDED)

    # Action 2: Add item 2 (separate intentional action in same or different task)
    attempt2, cached2 = ledger.prepare_action(
        task_id="task_cart_1",
        step_name="add_second_coffee",
        tool_name="browser_interact",
        arguments=args,
    )
    # Distinct step_name produces distinct idempotency_key
    assert cached2 is False
    assert attempt2.id != attempt1.id
    assert attempt2.attempt_number == 1
    assert attempt2.idempotency_key != attempt1.idempotency_key
    assert attempt2.arguments_hash == attempt1.arguments_hash


# ---------------------------------------------------------------------------
# 3. Concurrent Attempts Using the Same Idempotency Key
# ---------------------------------------------------------------------------

def test_concurrent_attempts_same_idempotency_key(ledger: ActionLedger) -> None:
    """Verify race condition handling: concurrent duplicate executions are rejected."""
    task_id = "task_race_1"
    step_name = "pay_bill"
    args = {"amount": 50.0}

    results: List[Tuple[ActionAttemptRecord, bool]] = []
    conflicts: List[str] = []
    lock = threading.Lock()

    def try_prepare() -> None:
        try:
            res = ledger.prepare_action(
                task_id=task_id,
                step_name=step_name,
                tool_name="payment_tool",
                arguments=args,
                initial_status=ActionLedgerStatus.EXECUTING,
            )
            with lock:
                results.append(res)
        except ActionLedgerConflictError as ce:
            with lock:
                conflicts.append(str(ce))

    threads = [threading.Thread(target=try_prepare) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread succeeds in entering EXECUTING state
    assert len(results) == 1
    assert len(conflicts) == 4


# ---------------------------------------------------------------------------
# 4. Crash After Marking an Action Executing
# ---------------------------------------------------------------------------

def test_crash_after_marking_action_executing(temp_db: Path) -> None:
    """Verify that process termination while executing marks action outcome_unknown without re-execution."""
    ledger1 = ActionLedger(db_path=temp_db, auto_recover=False)
    att1, _ = ledger1.prepare_action(
        task_id="task_crash_1",
        step_name="transfer_funds",
        tool_name="bank_transfer",
        arguments={"recipient": "acc_99", "amount": 100},
        initial_status=ActionLedgerStatus.EXECUTING,
    )
    assert att1.status == ActionLedgerStatus.EXECUTING

    # Simulate restart by instantiating new ActionLedger with auto_recover=True
    ledger2 = ActionLedger(db_path=temp_db, auto_recover=True)
    recovered = ledger2.get_attempt(att1.id)

    assert recovered is not None
    assert recovered.status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "NOT automatically re-executed" in recovered.status_message
    assert "Preserving uncertainty" in recovered.reconciliation_notes


# ---------------------------------------------------------------------------
# 5. Crash After Simulated External Effect and Outcome Reconciliation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_crash_after_external_effect_and_reconciliation(temp_db: Path) -> None:
    """Verify read-only reconciliation resolves an outcome_unknown action after a crash."""
    ledger1 = ActionLedger(db_path=temp_db, auto_recover=False)
    att1, _ = ledger1.prepare_action(
        task_id="task_crash_2",
        step_name="navigate_checkout",
        tool_name="browser_interact",
        arguments={"action": "navigate", "url": "https://example.com/confirmation"},
        initial_status=ActionLedgerStatus.EXECUTING,
    )

    # Crash recovery puts it in OUTCOME_UNKNOWN
    ledger2 = ActionLedger(db_path=temp_db, auto_recover=True)
    recovered = ledger2.get_attempt(att1.id)
    assert recovered.status == ActionLedgerStatus.OUTCOME_UNKNOWN

    # Mock browser controller observing that navigation did take effect
    mock_cdp = MagicMock(spec=CDPClient)
    browser = BrowserAutomationController(cdp_client=mock_cdp)

    # Mock read-only observation showing target URL reached
    browser.observe_page = AsyncMock(
        return_value={
            "success": True,
            "observation": {
                "url": "https://example.com/confirmation",
                "title": "Confirmation Page",
                "visible_text": "Your order #555 has been confirmed",
            },
        }
    )

    status, evidence = await browser.reconcile_action_outcome(
        action="navigate",
        target_url="https://example.com/confirmation",
    )
    assert status == ActionLedgerStatus.SUCCEEDED
    assert "target host" in evidence

    # Explicitly reconcile in ledger
    reconciled = ledger2.reconcile_outcome(att1.id, status, evidence)
    assert reconciled.status == ActionLedgerStatus.SUCCEEDED
    assert "Reconciled via read-only observation" in reconciled.reconciliation_notes


# ---------------------------------------------------------------------------
# 6. Timeout with Unknown Outcome
# ---------------------------------------------------------------------------

def test_timeout_with_unknown_outcome(ledger: ActionLedger) -> None:
    """Verify timed-out actions are marked outcome_unknown rather than assumed failed."""
    att, _ = ledger.prepare_action(
        task_id="task_timeout_1",
        step_name="submit_form",
        tool_name="browser_interact",
        arguments={"action": "click", "selector": "#submit"},
        initial_status=ActionLedgerStatus.EXECUTING,
    )

    # Network timed out before response was received
    updated = ledger.update_status(
        att.id,
        ActionLedgerStatus.OUTCOME_UNKNOWN,
        status_message="HTTP request timed out after 10s without confirmation",
        error_message="Gateway timeout",
    )
    assert updated.status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert updated.error_message == "Gateway timeout"

    # Attempting to re-prepare the same logical action raises ActionUncertaintyError
    with pytest.raises(ActionUncertaintyError) as exc_info:
        ledger.prepare_action(
            task_id="task_timeout_1",
            step_name="submit_form",
            tool_name="browser_interact",
            arguments={"action": "click", "selector": "#submit"},
        )
    assert "unknown outcome" in str(exc_info.value)
    assert "cannot retry without reconciliation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Read-Only Reconciliation Rules
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_read_only_reconciliation_rules() -> None:
    """Verify reconciliation distinguishes success signals, error signals, and ambiguous states."""
    mock_cdp = MagicMock(spec=CDPClient)
    browser = BrowserAutomationController(cdp_client=mock_cdp)

    # 1. Success Evidence
    browser.observe_page = AsyncMock(
        return_value={
            "success": True,
            "observation": {
                "url": "https://example.com/done",
                "title": "Success",
                "visible_text": "Transfer Complete. Thank you for your business.",
            },
        }
    )
    status_succ, _ = await browser.reconcile_action_outcome(action="click")
    assert status_succ == ActionLedgerStatus.SUCCEEDED

    # 2. Failure Evidence
    browser.observe_page = AsyncMock(
        return_value={
            "success": True,
            "observation": {
                "url": "https://example.com/checkout",
                "title": "Checkout",
                "visible_text": "Error: Insufficient funds. Payment declined.",
            },
        }
    )
    status_fail, _ = await browser.reconcile_action_outcome(action="click")
    assert status_fail == ActionLedgerStatus.FAILED

    # 3. Ambiguous State -> outcome_unknown
    browser.observe_page = AsyncMock(
        return_value={
            "success": True,
            "observation": {
                "url": "https://example.com/checkout",
                "title": "Checkout",
                "visible_text": "Items in cart: 1. Click to buy.",
            },
        }
    )
    status_ambig, _ = await browser.reconcile_action_outcome(action="click")
    assert status_ambig == ActionLedgerStatus.OUTCOME_UNKNOWN


# ---------------------------------------------------------------------------
# 8. Approval Authorization Separated from Idempotency
# ---------------------------------------------------------------------------

def test_approval_authorization_separated_from_idempotency(temp_db: Path) -> None:
    """Verify authorization boundary: an idempotency key cannot authorize execution without a token."""
    approval_mgr = ActionApprovalManager(db_path=temp_db)
    ledger = ActionLedger(db_path=temp_db, auto_recover=False)

    args = {"target": "user_42", "action": "delete"}
    pending = approval_mgr.create_pending_action(
        tool_name="user_admin",
        action_type="delete_user",
        target="user_42",
        arguments=args,
        description="Delete account user_42",
    )

    # Presenting an idempotency key without approval fails authorization
    ok, err = approval_mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="user_admin",
        arguments=args,
        approval_token="fabricated_token_12345",
    )
    assert ok is False

    # Legitimate approval
    ok_appr, _, token = approval_mgr.approve_action(pending.action_id, channel="trusted_ui")
    assert ok_appr is True and token is not None

    # Tampered arguments rejected
    tampered_args = {"target": "user_42", "action": "delete", "extra": "evil"}
    ok_tamp, err_tamp = approval_mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="user_admin",
        arguments=tampered_args,
        approval_token=token,
    )
    assert ok_tamp is False
    assert "rejected" in err_tamp.lower() or "not valid" in err_tamp.lower()


# ---------------------------------------------------------------------------
# 9. Restart Invalidation of Stale Approvals
# ---------------------------------------------------------------------------

def test_restart_invalidation_of_stale_approvals(temp_db: Path) -> None:
    """Verify unexecuted approved actions in the ledger are cancelled on restart."""
    approval_mgr1 = ActionApprovalManager(db_path=temp_db)
    ledger1 = ActionLedger(db_path=temp_db, auto_recover=False)

    pending = approval_mgr1.create_pending_action(
        tool_name="bank_tool",
        action_type="transfer",
        target="acc_1",
        arguments={"acc": "acc_1"},
        description="Transfer money",
    )
    approval_mgr1.approve_action(pending.action_id)

    att, _ = ledger1.prepare_action(
        task_id="task_restart_1",
        step_name="step_transfer",
        tool_name="bank_tool",
        arguments={"acc": "acc_1"},
        action_id=pending.action_id,
        initial_status=ActionLedgerStatus.APPROVED,
    )
    assert att.status == ActionLedgerStatus.APPROVED

    # Reinitialize with recovery
    ledger2 = ActionLedger(db_path=temp_db, auto_recover=True)
    rec = ledger2.get_attempt(att.id)
    assert rec.status == ActionLedgerStatus.CANCELLED
    assert "cancelled on restart" in rec.status_message


# ---------------------------------------------------------------------------
# 10. No Blind Replay of Uncertain Consequential Actions
# ---------------------------------------------------------------------------

def test_no_blind_replay_of_uncertain_actions(ledger: ActionLedger) -> None:
    """Verify that an action with outcome_unknown blocks new attempts until reconciled."""
    att, _ = ledger.prepare_action(
        task_id="task_uncertain_1",
        step_name="buy_shares",
        tool_name="stock_broker",
        arguments={"symbol": "AAPL", "quantity": 10},
        initial_status=ActionLedgerStatus.EXECUTING,
    )
    ledger.update_status(att.id, ActionLedgerStatus.OUTCOME_UNKNOWN)

    # Attempting to call prepare_action again fails
    with pytest.raises(ActionUncertaintyError):
        ledger.prepare_action(
            task_id="task_uncertain_1",
            step_name="buy_shares",
            tool_name="stock_broker",
            arguments={"symbol": "AAPL", "quantity": 10},
        )

    # Cannot transition out of outcome_unknown directly to executing
    with pytest.raises(InvalidActionStateTransitionError):
        ledger.update_status(att.id, ActionLedgerStatus.EXECUTING)

    # Only reconcile_outcome can transition to succeeded or failed
    reconciled = ledger.reconcile_outcome(
        att.id,
        ActionLedgerStatus.FAILED,
        evidence="Stock exchange logs show order 99 was rejected due to limit price",
    )
    assert reconciled.status == ActionLedgerStatus.FAILED

    # Once conclusively marked failed, a subsequent retry (attempt #2) is permitted
    retry_att, is_cached = ledger.prepare_action(
        task_id="task_uncertain_1",
        step_name="buy_shares",
        tool_name="stock_broker",
        arguments={"symbol": "AAPL", "quantity": 10},
    )
    assert is_cached is False
    assert retry_att.attempt_number == 2


# ---------------------------------------------------------------------------
# 11. Database Failures and Transaction Rollback
# ---------------------------------------------------------------------------

def test_database_failures_and_transaction_rollback(tmp_path: Path) -> None:
    """Ensure database errors fail closed and raise ActionLedgerError."""
    db_path = tmp_path / "corrupt_ledger.db"
    ledger = ActionLedger(db_path=db_path, auto_recover=False)

    with ledger._get_connection() as conn:
        conn.execute("DROP TABLE action_ledger;")

    with pytest.raises(ActionLedgerError):
        ledger.prepare_action(
            task_id="task_err",
            step_name="step_err",
            tool_name="test",
            arguments={},
        )


# ---------------------------------------------------------------------------
# 12. End-to-End Runtime Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_runtime_end_to_end_deduplication(tmp_path: Path) -> None:
    """Verify AgentRuntime deduplicates repeated affirmative confirmation turns."""
    db_path = tmp_path / "runtime_idemp.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    conv_storage = ConversationStorage(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path, auto_recover=False)

    class TransferArgs(BaseModel):
        target: str
        action_id: Optional[str] = None
        approval_token: Optional[str] = None
        user_confirmed: Optional[bool] = False

    registry = ToolRegistry()
    execution_counter = 0

    async def transfer_money(target: str, **kwargs: Any) -> Dict[str, Any]:
        nonlocal execution_counter
        user_confirmed = kwargs.get("user_confirmed", False)
        approval_token = kwargs.get("approval_token")

        if not user_confirmed and not approval_token:
            action = approval_manager.create_pending_action(
                tool_name="transfer_money",
                action_type="wire_transfer",
                target=target,
                arguments={"target": target},
                description=f"Wire 500 dollars to {target}",
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": action.action_id,
                "consequential_reason": "Wire transfers are permanent",
            }

        # Actual execution
        execution_counter += 1
        return {"success": True, "message": f"Successfully wired $500 to {target}"}

    registry.register(
        ToolDefinition(
            name="transfer_money",
            description="Wire transfer money",
            parameters_schema=TransferArgs.model_json_schema(),
            handler=transfer_money,
            args_model=TransferArgs,
            retry_safety=RetrySafety.NON_IDEMPOTENT,
        )
    )

    mock_client = AsyncMock()
    runtime = AgentRuntime(
        client=mock_client,
        registry=registry,
        conversation_storage=conv_storage,
        approval_manager=approval_manager,
        task_storage=task_storage,
        action_ledger=action_ledger,
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
                            "id": "call_wire",
                            "function": {"name": "transfer_money", "arguments": {"target": "acc_100"}},
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Wire transfer requires confirmation. Proceed?",
                }
            },
        ]
    )

    res1 = await runtime.run("Wire 500 to acc_100", conversation_id="conv_wire_1")
    assert res1.task_id is not None
    assert execution_counter == 0

    # Turn 2: User says "Yes, proceed"
    res2 = await runtime.run("Yes, proceed", conversation_id="conv_wire_1")
    assert "Confirmed and executed" in res2.final_response
    assert execution_counter == 1

    # Turn 3: User repeats "Yes, proceed" (duplicate turn or replayed prompt)
    res3 = await runtime.run("Yes, proceed", conversation_id="conv_wire_1")
    assert "already executed and verified" in res3.final_response
    # Tool was NOT called a second time!
    assert execution_counter == 1
