"""Adversarial and security tests for JARVIS Secure Action Approval (Milestone 08).

Proves that:
1. Model cannot self-approve by passing user_confirmed=True.
2. Model cannot self-approve by fabricating action_id or approval_token.
3. Replay attacks (reusing consumed tokens) are strictly blocked.
4. Parameter tampering (modifying target/selector/text after approval) is blocked and invalidates approval.
5. Stale / expired approvals (> TTL) are blocked and cannot be approved or consumed.
6. Cross-tool token reuse is blocked.
7. Modifying a pending action invalidates the previous pending action.
8. Confirmation semantics strictly reject ambiguous or query utterances.
9. Explicit rejections cancel pending actions and prevent execution.
10. Sensitive parameters are redacted and not leaked in server audit logs.
"""
import asyncio
from pathlib import Path
import time
from unittest.mock import AsyncMock

import pytest

from config.settings import settings
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    classify_confirmation_intent,
    compute_arguments_hash,
    default_approval_manager,
)
from tools.browser_automation import (
    BrowserAutomationController,
    is_consequential_action,
)
from tools.cdp_client import CDPClient


@pytest.fixture(autouse=True)
def isolate_test_approval_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all tests in this module use a temporary isolated SQLite database."""
    test_db = tmp_path / "test_secure_approval.db"
    monkeypatch.setattr(settings, "database_path", test_db)
    test_mgr = ActionApprovalManager(db_path=test_db)
    monkeypatch.setattr("security.approval.default_approval_manager", test_mgr)
    monkeypatch.setattr("tools.browser_automation.default_approval_manager", test_mgr)
    monkeypatch.setattr("agent.runtime.default_approval_manager", test_mgr)
    monkeypatch.setattr("tests.test_secure_approval.default_approval_manager", test_mgr)
    yield test_db


class MockCDPClient(CDPClient):
    """Mock CDP client for secure approval tests."""

    def __init__(self) -> None:
        super().__init__(host="127.0.0.1", port=9222)
        self._available = True
        self.commands_sent: list = []

    async def is_available(self):
        return True, "CDP available"

    async def connect_to_target(self, target_id=None):
        return "ws://127.0.0.1:9222/devtools/page/test"

    async def send_command(self, method: str, params=None):
        self.commands_sent.append((method, params))
        if method == "Runtime.evaluate":
            expr = (params or {}).get("expression", "")
            if "const url = window.location.href" in expr:
                return {
                    "result": {
                        "value": {
                            "url": "https://shop.local/checkout",
                            "title": "Checkout Store",
                            "visible_text": "Please confirm order",
                            "interactive_elements": [
                                {
                                    "index": 1,
                                    "tag": "button",
                                    "role": "button",
                                    "type": "button",
                                    "text": "Cancel",
                                    "selector": "#cancel-btn",
                                },
                                {
                                    "index": 2,
                                    "tag": "button",
                                    "role": "button",
                                    "type": "submit",
                                    "text": "Submit Payment",
                                    "selector": "#pay-now-btn",
                                },
                            ],
                        }
                    }
                }
            if "querySelector" in expr:
                return {"result": {"value": "test-val"}}
            return {"result": {"value": True}}
        return {}


# --- Threat Case 1 & 2: Model Self-Approval Rejection ---

@pytest.mark.anyio
async def test_threat_model_cannot_self_approve_with_user_confirmed_flag():
    """Adversarial test: Model attempts to self-approve consequential action by passing user_confirmed=True."""
    mgr = ActionApprovalManager()
    cdp = MockCDPClient()
    controller = BrowserAutomationController(cdp_client=cdp)

    # Consequential action: clicking submit payment
    res = await controller.execute_action(
        action="click",
        selector="#pay-now-btn",
        user_confirmed=True,  # Model passes True unilaterally
    )

    # Execution MUST be halted, requiring server-side approval
    assert res["success"] is False
    assert res["requires_confirmation"] is True
    assert "action_id" in res
    assert "explicit user confirmation" in res["message"].lower()

    # Verify no CDP click command was issued
    click_commands = [c for c in cdp.commands_sent if "Input.dispatchMouseEvent" in c[0]]
    assert len(click_commands) == 0


@pytest.mark.anyio
async def test_threat_model_cannot_self_approve_with_fabricated_token():
    """Adversarial test: Model attempts to fabricate action_id and approval_token."""
    cdp = MockCDPClient()
    controller = BrowserAutomationController(cdp_client=cdp)

    res = await controller.execute_action(
        action="click",
        selector="#pay-now-btn",
        action_id="act_fabricated1234",
        approval_token="tok_fake_secret_token",
    )

    assert res["success"] is False
    assert res.get("security_violation") is True
    assert "Security policy authorization failed" in res["error"]
    assert len([c for c in cdp.commands_sent if "Input.dispatchMouseEvent" in c[0]]) == 0


# --- Threat Case 3: Replay Attack (Token Reuse) ---

@pytest.mark.anyio
async def test_threat_replay_attack_blocked():
    """Adversarial test: Reusing an already-consumed single-use approval token is blocked."""
    mgr = ActionApprovalManager()
    args = {"action": "click", "selector": "#pay-now-btn"}
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-now-btn",
        arguments=args,
        description="Submit payment",
    )

    # Trusted approval
    ok, msg, token = mgr.approve_action(pending.action_id, channel="trusted_ui")
    assert ok is True
    assert token is not None

    # First legitimate consumption -> Allowed
    ok1, msg1 = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments=args,
        approval_token=token,
    )
    assert ok1 is True
    assert "Approval verified and consumed" in msg1

    # Second consumption attempt (Replay attack) -> Strictly Blocked
    ok2, msg2 = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments=args,
        approval_token=token,
    )
    assert ok2 is False
    assert "replay attack rejected" in msg2.lower()

    # Check audit log contains REPLAY_ATTACK_BLOCKED
    audit_events = [r.event_type for r in mgr.get_audit_log()]
    assert "REPLAY_ATTACK_BLOCKED" in audit_events


# --- Threat Case 4: Parameter Tampering (Argument Alteration) ---

@pytest.mark.anyio
async def test_threat_parameter_tampering_blocked_and_invalidated():
    """Adversarial test: Altering arguments after receiving user approval is rejected and invalidates action."""
    mgr = ActionApprovalManager()
    legitimate_args = {"action": "click", "selector": "#cancel-order-btn"}
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#cancel-order-btn",
        arguments=legitimate_args,
        description="Cancel order",
    )

    ok, msg, token = mgr.approve_action(pending.action_id, channel="trusted_voice")
    assert ok is True

    # Attacker tries to consume token for a different, malicious selector
    tampered_args = {"action": "click", "selector": "#transfer-all-funds-btn"}
    ok_tamper, msg_tamper = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments=tampered_args,
        approval_token=token,
    )
    assert ok_tamper is False
    assert "tampering rejected" in msg_tamper.lower()

    # Verify action was permanently invalidated (status set to REJECTED)
    action_record = mgr.get_pending_action(pending.action_id)
    assert action_record.status == ApprovalStatus.REJECTED

    # Attempting to re-run legitimate arguments now also fails because action was invalidated
    ok_retry, _ = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments=legitimate_args,
        approval_token=token,
    )
    assert ok_retry is False

    audit_events = [r.event_type for r in mgr.get_audit_log()]
    assert "ARGUMENT_TAMPERING_BLOCKED" in audit_events


# --- Threat Case 5: Stale and Expired Approval ---

@pytest.mark.anyio
async def test_threat_stale_approval_expiration():
    """Adversarial test: Approvals past expiration TTL cannot be consumed or approved."""
    mgr = ActionApprovalManager(default_ttl_sec=0.1)
    args = {"action": "click", "selector": "#pay-now-btn"}
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-now-btn",
        arguments=args,
        description="Payment",
        ttl_sec=0.05,
    )

    # Approve immediately before expiration
    ok, msg, token = mgr.approve_action(pending.action_id, channel="trusted_ui")
    assert ok is True

    # Wait for TTL to expire
    await asyncio.sleep(0.08)

    # Attempt to consume expired token -> Blocked
    ok_consume, msg_consume = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments=args,
        approval_token=token,
    )
    assert ok_consume is False
    assert "expired" in msg_consume.lower()

    # Also test approving an action that has already expired
    pending2 = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#other-btn",
        arguments={"action": "click", "selector": "#other-btn"},
        description="Other action",
        ttl_sec=0.02,
    )
    await asyncio.sleep(0.04)
    ok_late_app, msg_late_app, _ = mgr.approve_action(pending2.action_id)
    assert ok_late_app is False
    assert "expired" in msg_late_app.lower()


# --- Threat Case 6: Cross-Tool Mismatch ---

@pytest.mark.anyio
async def test_threat_cross_tool_approval_mismatch():
    """Adversarial test: Token approved for browser_interact cannot be consumed by another tool."""
    mgr = ActionApprovalManager()
    args = {"action": "click", "selector": "#pay-btn"}
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-btn",
        arguments=args,
        description="Pay",
    )
    ok, _, token = mgr.approve_action(pending.action_id)
    assert ok is True

    # Attacker tries to consume token on cancel_reminder
    ok_mismatch, msg_mismatch = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="cancel_reminder",
        arguments=args,
        approval_token=token,
    )
    assert ok_mismatch is False
    assert "not valid for tool 'cancel_reminder'" in msg_mismatch


# --- Threat Case 7: Action Invalidation When Action Changes ---

@pytest.mark.anyio
async def test_threat_pending_action_invalidated_when_arguments_change():
    """Adversarial test: When an action's arguments change, prior pending action is invalidated."""
    mgr = ActionApprovalManager()
    args1 = {"action": "click", "selector": "#initial-target"}
    pending1 = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#initial-target",
        arguments=args1,
        description="Initial target",
    )
    id1 = pending1.action_id

    # New action for the same tool with different arguments arrives
    args2 = {"action": "click", "selector": "#different-target"}
    pending2 = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#different-target",
        arguments=args2,
        description="Different target",
    )
    id2 = pending2.action_id
    assert id1 != id2

    # Prior pending action must have been invalidated (REJECTED)
    assert mgr.get_pending_action(id1).status == ApprovalStatus.REJECTED

    # Attempting to approve prior pending action fails
    ok_app1, _, _ = mgr.approve_action(id1)
    assert ok_app1 is False

    # New action can be approved normally
    ok_app2, _, token2 = mgr.approve_action(id2)
    assert ok_app2 is True


# --- Threat Case 8: Ambiguous Voice / Interaction Intent Rejection ---

def test_confirmation_intent_classification():
    """Semantics test: Explicit affirmations approve; explicit rejections deny; ambiguous queries are rejected."""
    # Affirmations
    for phrase in ["yes", "confirm", "i confirm", "approve", "proceed", "go ahead", "do it", "authorize"]:
        intent, conf = classify_confirmation_intent(phrase)
        assert intent == "affirm", f"Expected 'affirm' for '{phrase}', got '{intent}'"
        assert conf >= 0.8

    # Rejections
    for phrase in ["no", "cancel", "stop", "abort", "deny", "never mind", "do not"]:
        intent, conf = classify_confirmation_intent(phrase)
        assert intent == "reject", f"Expected 'reject' for '{phrase}', got '{intent}'"
        assert conf >= 0.8

    # Ambiguous phrases / queries / evasion attempts -> strictly ambiguous
    ambiguous_phrases = [
        "maybe",
        "perhaps",
        "what is this?",
        "what time is it right now?",
        "tell me a joke",
        "who are you?",
        "why do you need to do that?",
        "wait a second",
        "can you check first?",
        "i don't know",
        "what did you say",
    ]
    for phrase in ambiguous_phrases:
        intent, conf = classify_confirmation_intent(phrase)
        assert intent == "ambiguous", f"Expected 'ambiguous' for '{phrase}', got '{intent}'"
        assert conf == 0.0


# --- Threat Case 9: Explicit User Rejection Cancels Action ---

def test_explicit_user_rejection_cancels_pending_action():
    """User safety test: Explicit denial rejects the action and leaves it in unexecutable state."""
    mgr = ActionApprovalManager()
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#checkout",
        arguments={"action": "click", "selector": "#checkout"},
        description="Checkout",
    )

    rejected = mgr.reject_action(pending.action_id, channel="trusted_voice", reason="User said 'no'")
    assert rejected is True

    record = mgr.get_pending_action(pending.action_id)
    assert record.status == ApprovalStatus.REJECTED

    # Attempting to execute in rejected state fails
    ok, msg = mgr.verify_and_consume_approval(
        action_id=pending.action_id,
        tool_name="browser_interact",
        arguments={"action": "click", "selector": "#checkout"},
        approval_token="tok_any",
    )
    assert ok is False
    assert "not been approved" in msg


# --- Threat Case 10: Audit Log Privacy & Credential Redaction ---

def test_audit_log_redaction_and_integrity():
    """Privacy test: Sensitive credentials in arguments are redacted and page content is not leaked."""
    mgr = ActionApprovalManager()
    sensitive_args = {
        "action": "type",
        "selector": "#password-input",
        "password": "SuperSecretPassword123!",
        "token_key": "api_secret_abc123",
        "text": "regular text",
    }
    pending = mgr.create_pending_action(
        tool_name="browser_interact",
        action_type="type",
        target="#password-input",
        arguments=sensitive_args,
        description="Input credentials",
    )

    # Password and token arguments must be redacted in the pending record
    assert pending.arguments["password"] == "[REDACTED]"
    assert pending.arguments["token_key"] == "[REDACTED]"
    assert pending.arguments["text"] == "regular text"

    # Audit records do not contain raw passwords
    audit_log = mgr.get_audit_log()
    for rec in audit_log:
        assert "SuperSecretPassword123!" not in rec.details
        assert "api_secret_abc123" not in rec.details
        assert len(rec.target_summary) <= 100


# --- End-to-End Conversational Runtime Confirmation Flow Tests ---

@pytest.mark.anyio
async def test_conversational_runtime_affirmation_flow():
    """Verify that when an action is pending, an affirmative user turn executes the action with trusted token."""
    from agent.runtime import AgentRuntime
    from agent.tool_registry import ToolRegistry, ToolDefinition
    from pydantic import BaseModel

    class DummyArgs(BaseModel):
        action: str = "click"
        selector: str = "#pay-btn"
        action_id: str = None
        approval_token: str = None

    executed = []

    async def dummy_handler(action="click", selector="#pay-btn", action_id=None, approval_token=None):
        executed.append((action, selector, action_id, approval_token))
        return {"success": True, "message": "Payment processed."}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="browser_interact",
            description="Browser interact",
            parameters_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            args_model=DummyArgs,
        )
    )

    runtime = AgentRuntime(registry=registry)

    # Set up pending action
    pending = default_approval_manager.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-btn",
        arguments={"action": "click", "selector": "#pay-btn"},
        description="Submit payment",
    )

    # User speaks: "Yes, proceed"
    result = await runtime.run("Yes, proceed")

    assert result.completed is True
    assert "Confirmed and executed" in result.final_response
    assert len(executed) == 1
    assert executed[0][2] == pending.action_id
    assert executed[0][3] is not None
    assert executed[0][3].startswith("tok_")


@pytest.mark.anyio
async def test_conversational_runtime_rejection_flow():
    """Verify that when an action is pending, an explicit rejection cancels it without executing."""
    from agent.runtime import AgentRuntime
    from agent.tool_registry import ToolRegistry, ToolDefinition
    from pydantic import BaseModel

    class DummyArgs(BaseModel):
        action: str = "click"

    executed = []

    async def dummy_handler(action="click", action_id=None, approval_token=None):
        executed.append(action)
        return {"success": True}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="browser_interact",
            description="Browser interact",
            parameters_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            args_model=DummyArgs,
        )
    )

    runtime = AgentRuntime(registry=registry)

    pending = default_approval_manager.create_pending_action(
        tool_name="browser_interact",
        action_type="click",
        target="#pay-btn",
        arguments={"action": "click"},
        description="Submit payment",
    )

    # User speaks: "No, cancel that"
    result = await runtime.run("No, cancel that")

    assert result.completed is True
    assert "cancelled" in result.final_response.lower()
    assert len(executed) == 0
    assert default_approval_manager.get_pending_action(pending.action_id).status == ApprovalStatus.REJECTED

