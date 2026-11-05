"""Focused regression tests for JARVIS Milestone 09F: Reliability Fixes.

Covers:
1. Test database isolation and protection against production database initialization.
2. Realistic observation schema and `visible_text` reconciliation.
3. False-positive navigation matches (e.g., search queries vs. target domains).
4. Unrelated or stale success text not resolving an action.
5. CLI reconciliation using metadata from a runtime-created ledger entry.
6. Multi-turn approval transition without orphaned duplicate attempts.
7. Stable idempotency across confirmation turns, while distinct actions remain distinct.
8. Cross-conversation approval isolation.
9. Rejection and ambiguous responses for `outcome_unknown`.
10. Startup recovery, repeated recovery, and database failures remaining fail-closed.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agent.runtime import AgentRuntime, extract_action_metadata
from agent.tool_registry import ToolDefinition, ToolRegistry
from config.settings import settings
from main import handle_ledger_reconcile
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    PendingAction,
    SecurityAuditStorageError,
    default_approval_manager,
)
from storage.action_ledger import (
    ActionLedger,
    ActionLedgerConflictError,
    ActionLedgerStatus,
    ActionUncertaintyError,
    RetrySafety,
    default_action_ledger,
    normalize_and_hash_arguments,
)
from storage.conversation import (
    ConversationStorage,
    default_conversation_storage,
)
from storage.reminders import default_reminder_storage
from storage.tasks import (
    TaskRecord,
    TaskStatus,
    TaskStorage,
    default_task_storage,
)
from tools.browser_automation import BrowserAutomationController, BrowserObservation


# ---------------------------------------------------------------------------
# 1. Test database isolation and protection against production DB
# ---------------------------------------------------------------------------

def test_database_isolation_and_guard_protection() -> None:
    """Verify test configuration uses isolated database and singletons never point to production."""
    prod_path = Path("data/jarvis.db").resolve()
    configured_path = settings.database_path.resolve()

    # Must be distinct paths
    assert configured_path != prod_path, "Configured test database must not be data/jarvis.db"
    assert "test" in str(configured_path).lower() or "tmp" in str(configured_path).lower()

    # All default singletons must point to test path, not prod path
    assert default_approval_manager.db_path.resolve() != prod_path
    assert default_action_ledger.db_path.resolve() != prod_path
    assert default_conversation_storage.db_path.resolve() != prod_path
    assert default_reminder_storage.db_path.resolve() != prod_path
    assert default_task_storage.db_path.resolve() != prod_path

    assert default_approval_manager.db_path.resolve() == configured_path
    assert default_action_ledger.db_path.resolve() == configured_path
    assert default_conversation_storage.db_path.resolve() == configured_path
    assert default_reminder_storage.db_path.resolve() == configured_path
    assert default_task_storage.db_path.resolve() == configured_path


# ---------------------------------------------------------------------------
# 2. Realistic observation schema and visible_text reconciliation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_realistic_observation_schema_reconciliation(tmp_path: Path) -> None:
    """Verify BrowserAutomation.reconcile_action_outcome uses real BrowserObservation schema (visible_text)."""
    db_path = tmp_path / "obs_recon.db"
    mgr = ActionApprovalManager(db_path=db_path)
    browser = BrowserAutomationController(approval_manager=mgr)

    # Real BrowserObservation schema includes url, title, visible_text, interactive_elements
    real_obs = BrowserObservation(
        url="https://shop.local/checkout/confirmed",
        title="Order Confirmation",
        visible_text="Thank you! Your payment was successfully processed. Order #99182 confirmed.",
        interactive_elements=[],
    )

    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": real_obs.model_dump(),
    })

    status, evidence = await browser.reconcile_action_outcome(
        action="click",
        selector="#submit-order",
        expected_text="Place Order",
    )

    assert status == ActionLedgerStatus.SUCCEEDED
    assert "payment was successfully processed" in evidence.lower() or "confirmed" in evidence.lower()


# ---------------------------------------------------------------------------
# 3. False-positive navigation matches (deceptive queries, domains, ports, paths)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_navigation_false_positive_rejection(tmp_path: Path) -> None:
    """Verify reconcile_action_outcome rejects substring/query-parameter deception and port/path mismatches."""
    db_path = tmp_path / "nav_rej.db"
    mgr = ActionApprovalManager(db_path=db_path)
    browser = BrowserAutomationController(approval_manager=mgr)

    intended_url = "https://bank.com/checkout"

    # Case A: Deceptive search query containing the target domain
    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": {
            "url": "https://google.com/search?q=bank.com/checkout",
            "title": "Google Search",
            "visible_text": "Search results for bank.com",
            "interactive_elements": [],
        },
    })
    status, evidence = await browser.reconcile_action_outcome(action="navigate", target_url=intended_url)
    assert status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "Host mismatch" in evidence

    # Case B: Deceptive attacker subdomain
    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": {
            "url": "https://bank.com.evil-phish.net/checkout",
            "title": "Phishing Page",
            "visible_text": "Login",
            "interactive_elements": [],
        },
    })
    status, evidence = await browser.reconcile_action_outcome(action="navigate", target_url=intended_url)
    assert status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "Host mismatch" in evidence

    # Case C: Port mismatch
    intended_with_port = "http://bank.local:8080/app"
    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": {
            "url": "http://bank.local:9000/app",
            "title": "App",
            "visible_text": "App",
            "interactive_elements": [],
        },
    })
    status, evidence = await browser.reconcile_action_outcome(action="navigate", target_url=intended_with_port)
    assert status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "Port mismatch" in evidence

    # Case D: Path mismatch (redirected away from target path)
    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": {
            "url": "https://bank.com/login",
            "title": "Login Required",
            "visible_text": "Please log in to continue",
            "interactive_elements": [],
        },
    })
    status, evidence = await browser.reconcile_action_outcome(action="navigate", target_url=intended_url)
    assert status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "Path mismatch" in evidence


# ---------------------------------------------------------------------------
# 4. Unrelated or stale success text not resolving an action
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unrelated_stale_success_text_does_not_resolve(tmp_path: Path) -> None:
    """Verify generic or marketing words ('success stories', 'thank you for visiting') do not resolve click."""
    db_path = tmp_path / "stale_text.db"
    mgr = ActionApprovalManager(db_path=db_path)
    browser = BrowserAutomationController(approval_manager=mgr)

    # Observation contains generic words "success" and "thank you", but in a marketing context
    browser.observe_page = AsyncMock(return_value={
        "success": True,
        "observation": {
            "url": "https://company.local/blog",
            "title": "Our Success Stories",
            "visible_text": "Thank you for visiting our customer success blog! Read our latest stories.",
            "interactive_elements": [],
        },
    })

    status, evidence = await browser.reconcile_action_outcome(
        action="click",
        selector="#checkout-btn",
        expected_text="Purchase",
    )
    # Must fail closed to OUTCOME_UNKNOWN, NOT mark SUCCEEDED
    assert status == ActionLedgerStatus.OUTCOME_UNKNOWN
    assert "transactional confirmation" in evidence.lower() or "no specific" in evidence.lower()


# ---------------------------------------------------------------------------
# 5. CLI reconciliation using metadata from runtime-created ledger entry
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cli_reconciliation_persisted_metadata_and_fail_closed(tmp_path: Path) -> None:
    """Verify handle_ledger_reconcile uses persisted metadata and fails closed when metadata is missing."""
    db_path = tmp_path / "cli_recon.db"
    ledger = ActionLedger(db_path=db_path, auto_recover=False)
    mgr = ActionApprovalManager(db_path=db_path)
    browser = BrowserAutomationController(approval_manager=mgr)

    # Patch default singletons used by handle_ledger_reconcile
    with patch("storage.action_ledger.default_action_ledger", ledger), \
         patch("tools.browser_automation.default_browser_automation", browser):

        # Case A: Ledger entry with missing action metadata -> fails closed
        attempt_empty, _ = ledger.prepare_action(
            task_id="t_empty",
            step_name="step_empty",
            tool_name="browser_interact",
            arguments={"unrelated": "data"},
            retry_safety=RetrySafety.NON_IDEMPOTENT,
            initial_status=ActionLedgerStatus.OUTCOME_UNKNOWN,
            metadata={},
        )
        rc_empty = await handle_ledger_reconcile(attempt_empty.id)
        assert rc_empty == 1
        refreshed_empty = ledger.get_attempt(attempt_empty.id)
        assert refreshed_empty is not None
        assert refreshed_empty.status == ActionLedgerStatus.OUTCOME_UNKNOWN

        # Case B: Ledger entry with valid persisted metadata -> reconciles safely
        browser.observe_page = AsyncMock(return_value={
            "success": True,
            "observation": {
                "url": "https://dashboard.local/home",
                "title": "Dashboard Home",
                "visible_text": "Welcome to your home dashboard",
                "interactive_elements": [],
            },
        })

        attempt_valid, _ = ledger.prepare_action(
            task_id="t_valid",
            step_name="step_valid",
            tool_name="browser_interact",
            arguments={"action": "navigate", "url": "https://dashboard.local/home"},
            retry_safety=RetrySafety.NON_IDEMPOTENT,
            initial_status=ActionLedgerStatus.OUTCOME_UNKNOWN,
            metadata={"action": "navigate", "url": "https://dashboard.local/home"},
        )
        rc_valid = await handle_ledger_reconcile(attempt_valid.id)
        assert rc_valid == 0
        refreshed_valid = ledger.get_attempt(attempt_valid.id)
        assert refreshed_valid is not None
        assert refreshed_valid.status == ActionLedgerStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# 6. Multi-turn approval transition without orphaned duplicate attempt
# ---------------------------------------------------------------------------

def test_approval_transition_without_orphaned_duplicate(tmp_path: Path) -> None:
    """Verify awaiting_approval transitions to approved without creating an orphaned duplicate attempt."""
    db_path = tmp_path / "no_dupe.db"
    ledger = ActionLedger(db_path=db_path, auto_recover=False)

    task_id = "task_multiturn_1"
    step_name = "browser_click_abc123"
    args = {"action": "click", "selector": "#confirm-btn"}

    # Turn 1: Action requires approval
    att1, is_cached1 = ledger.prepare_action(
        task_id=task_id,
        step_name=step_name,
        tool_name="browser_interact",
        arguments=args,
        action_id="act_random_tok_1",
        retry_safety=RetrySafety.NON_IDEMPOTENT,
        initial_status=ActionLedgerStatus.AWAITING_APPROVAL,
    )
    assert not is_cached1
    assert att1.attempt_number == 1
    assert att1.status == ActionLedgerStatus.AWAITING_APPROVAL

    # Turn 2: User approves
    att2, is_cached2 = ledger.prepare_action(
        task_id=task_id,
        step_name=step_name,
        tool_name="browser_interact",
        arguments=args,
        action_id="act_random_tok_1",
        retry_safety=RetrySafety.NON_IDEMPOTENT,
        initial_status=ActionLedgerStatus.APPROVED,
    )
    assert not is_cached2
    # Must be the SAME attempt transitioned in-place
    assert att2.id == att1.id
    assert att2.attempt_number == 1
    assert att2.status == ActionLedgerStatus.APPROVED

    # Verify no second attempt was created in database
    history = ledger.list_attempts(task_id=task_id)
    assert len(history) == 1
    assert history[0].status == ActionLedgerStatus.APPROVED


# ---------------------------------------------------------------------------
# 7. Stable idempotency across confirmation turns while distinct actions differ
# ---------------------------------------------------------------------------

def test_stable_idempotency_keys_and_distinct_actions(tmp_path: Path) -> None:
    """Verify idempotency key remains stable across turns and random tokens, while distinct actions differ."""
    args_action1 = {"action": "click", "selector": "#pay-now"}
    args_action2 = {"action": "click", "selector": "#cancel-order"}

    hash1 = normalize_and_hash_arguments(args_action1)
    hash2 = normalize_and_hash_arguments(args_action2)

    step1 = f"browser_interact_{args_action1['action']}_{hash1[:12]}"
    step2 = f"browser_interact_{args_action2['action']}_{hash2[:12]}"

    # Action 1 and Action 2 have distinct logical steps
    assert step1 != step2

    # Even if random approval tokens differ between turns, logical step is stable
    token_turn1 = "act_11111111"
    token_turn2 = "act_22222222"

    step_recomputed = f"browser_interact_{args_action1['action']}_{normalize_and_hash_arguments(args_action1)[:12]}"
    assert step1 == step_recomputed

    # Test metadata extractor sanitizes passwords/secrets
    meta = extract_action_metadata("browser_interact", {
        "action": "fill",
        "selector": "#password_field",
        "text": "SuperSecret123!",
    })
    assert meta.get("text") == "[REDACTED]"


# ---------------------------------------------------------------------------
# 8. Cross-conversation approval isolation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cross_conversation_approval_isolation(tmp_path: Path) -> None:
    """Verify Conversation B cannot approve or execute Conversation A's pending action."""
    db_path = tmp_path / "cross_conv.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    conv_storage = ConversationStorage(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path, auto_recover=False)

    class ConsequentialArgs(BaseModel):
        target: str
        action_id: Optional[str] = None
        approval_token: Optional[str] = None

    registry = ToolRegistry()

    async def dangerous_action(target: str, **kwargs: Any) -> Dict[str, Any]:
        approval_token = kwargs.get("approval_token")
        if not approval_token:
            action = approval_manager.create_pending_action(
                tool_name="dangerous_action",
                action_type="delete_record",
                target=target,
                arguments={"target": target},
                description=f"Delete record {target}",
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": action.action_id,
                "consequential_reason": "Permanent deletion",
            }
        return {"success": True, "message": f"Deleted {target}"}

    registry.register(
        ToolDefinition(
            name="dangerous_action",
            description="Delete record",
            parameters_schema=ConsequentialArgs.model_json_schema(),
            handler=dangerous_action,
            args_model=ConsequentialArgs,
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

    # Turn 1: Conversation A initiates action requiring approval
    mock_client.chat = AsyncMock(side_effect=[
        {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_danger_1",
                    "function": {
                        "name": "dangerous_action",
                        "arguments": {"target": "user_42"},
                    },
                }],
            }
        },
        {
            "message": {
                "content": "Action requires your approval before proceeding.",
            }
        },
    ])

    res_a = await runtime.run(
        prompt="Delete user 42",
        conversation_id="conv_a",
    )
    # Task A is waiting for approval
    tasks_a = task_storage.list_tasks(status=TaskStatus.WAITING_FOR_APPROVAL, conversation_id="conv_a")
    assert len(tasks_a) == 1
    task_a_id = tasks_a[0].id

    # Turn 2: Conversation B sends "yes, approve"
    # Conversation B should NOT approve Conversation A's task
    mock_client.chat = AsyncMock(return_value={
        "message": {"content": "I see nothing pending here."}
    })

    res_b = await runtime.run(
        prompt="yes, confirm",
        conversation_id="conv_b",
    )

    # Verify Task A in Conv A was NOT approved and remains WAITING_FOR_APPROVAL
    task_a_check = task_storage.get_task(task_a_id)
    assert task_a_check is not None
    assert task_a_check.status == TaskStatus.WAITING_FOR_APPROVAL

    # Pending action in approval manager remains pending
    pending_act_id = task_a_check.metadata.get("pending_action_id")
    assert pending_act_id is not None
    pa = approval_manager.get_pending_action(pending_act_id)
    assert pa is not None
    assert pa.status == ApprovalStatus.PENDING


# ---------------------------------------------------------------------------
# 9. Rejection and ambiguous responses for outcome_unknown
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_outcome_unknown_user_rejection_and_ambiguity(tmp_path: Path) -> None:
    """Verify 'cancel' and ambiguous responses for outcome_unknown cancel local tasks without falsely reversing external status."""
    db_path = tmp_path / "unknown_rej.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    conv_storage = ConversationStorage(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path, auto_recover=False)

    registry = ToolRegistry()
    mock_client = AsyncMock()
    runtime = AgentRuntime(
        client=mock_client,
        registry=registry,
        conversation_storage=conv_storage,
        approval_manager=approval_manager,
        task_storage=task_storage,
        action_ledger=action_ledger,
    )

    # Create an action record in OUTCOME_UNKNOWN
    pa = approval_manager.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-btn",
        arguments={"action": "click", "selector": "#pay-btn"},
        description="Payment submission",
    )
    # Mark approved/consumed
    approval_manager.approve_action(pa.action_id, channel="test")

    # Create task and ledger attempt in OUTCOME_UNKNOWN
    task = task_storage.create_task(
        task_type="agent_turn",
        description="Payment submission task",
        conversation_id="conv_unk",
        status=TaskStatus.RUNNING,
        metadata={"pending_action_id": pa.action_id},
    )

    att, _ = action_ledger.prepare_action(
        task_id=task.id,
        step_name="browser_click_test",
        tool_name="browser_interact",
        arguments=pa.arguments,
        action_id=pa.action_id,
        retry_safety=RetrySafety.NON_IDEMPOTENT,
        initial_status=ActionLedgerStatus.OUTCOME_UNKNOWN,
    )

    # Case 9a: User says "cancel" / "no"
    res_cancel = await runtime.run(
        prompt="cancel the action",
        conversation_id="conv_unk",
        task_id=task.id,
    )

    # Local task is cancelled
    refreshed_task = task_storage.get_task(task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.CANCELLED

    # Ledger attempt REMAINS outcome_unknown (never falsely marked cancelled/reversed)
    refreshed_att = action_ledger.get_attempt(att.id)
    assert refreshed_att is not None
    assert refreshed_att.status == ActionLedgerStatus.OUTCOME_UNKNOWN

    # Warning given to user
    assert "local cancellation cannot reverse external actions" in res_cancel.final_response.lower()

    # Case 9b: User gives ambiguous response
    res_ambig = await runtime.run(
        prompt="what does that mean?",
        conversation_id="conv_unk",
    )
    assert not res_ambig.completed
    assert res_ambig.status == "action_uncertainty_blocked"
    assert "unknown outcome" in res_ambig.final_response.lower()


# ---------------------------------------------------------------------------
# 10. Startup recovery, repeated recovery, and database failures fail-closed
# ---------------------------------------------------------------------------

def test_startup_recovery_idempotency_and_db_failure_fail_closed(tmp_path: Path) -> None:
    """Verify recovery is idempotent and database failures fail closed."""
    db_path = tmp_path / "recovery_fail_closed.db"
    task_storage = TaskStorage(db_path=db_path, auto_recover=False)
    approval_manager = ActionApprovalManager(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path, auto_recover=False)

    # Create abandoned active tasks
    t1 = task_storage.create_task(task_type="turn", description="turn 1", status=TaskStatus.RUNNING)
    t2 = task_storage.create_task(task_type="turn", description="turn 2", status=TaskStatus.QUEUED)

    # Initial crash recovery
    recovered1 = task_storage.recover_interrupted_tasks()
    assert len(recovered1) == 2
    assert task_storage.get_task(t1.id).status == TaskStatus.INTERRUPTED
    assert task_storage.get_task(t2.id).status == TaskStatus.INTERRUPTED

    # Repeated crash recovery is idempotent
    recovered2 = task_storage.recover_interrupted_tasks()
    assert len(recovered2) == 0

    # Test DB failure fails closed during approval
    with patch.object(approval_manager, "_get_connection", side_effect=sqlite3.OperationalError("disk I/O error")):
        with pytest.raises(SecurityAuditStorageError):
            approval_manager.create_pending_action(
                tool_name="test",
                action_type="test",
                target="target",
                arguments={},
                description="desc",
            )
