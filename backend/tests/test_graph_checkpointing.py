"""Focused tests for Durable LangGraph Checkpointing and Resumption in JARVIS."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from agent.graph_runtime import LangGraphAgentRuntime
from agent.runtime import create_agent_runtime, ToolExecutionRecord
from agent.ollama_client import LLMClientProtocol, OllamaResponseError
from agent.tool_registry import ToolDefinition, create_default_registry
from storage.action_ledger import ActionLedger, ActionLedgerStatus, RetrySafety
from storage.checkpoint import CheckpointStorage
from storage.conversation import ConversationStorage
from storage.tasks import TaskStorage, TaskStatus
from security.approval import ActionApprovalManager


class MockCheckpointClient:
    """Mock LLM client fulfilling LLMClientProtocol for checkpoint tests."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None) -> None:
        self._responses = list(responses or [])
        self.call_history: List[Dict[str, Any]] = []

    def queue_response(self, response: Dict[str, Any]) -> None:
        self._responses.append(response)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.call_history.append({
            "messages": list(messages),
            "tools": tools,
            "options": options,
            "model": model,
        })
        if not self._responses:
            raise OllamaResponseError("No mock responses remaining.")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return next_resp


@pytest.fixture
def isolated_storage(tmp_path):
    """Provide isolated SQLite storage backends for testing."""
    db_path = tmp_path / "test_cp_storage.db"
    return {
        "conv": ConversationStorage(db_path=db_path),
        "task": TaskStorage(db_path=db_path),
        "ledger": ActionLedger(db_path=db_path),
        "approval": ActionApprovalManager(db_path=db_path),
    }


@pytest.mark.anyio
async def test_checkpoint_storage_lifecycle_and_reuse(tmp_path):
    """Verify CheckpointStorage manages connections, creates tables, and closes safely."""
    cp_db = tmp_path / "lifecycle_checkpoints.db"
    storage = CheckpointStorage(db_path=cp_db)

    # Initial state
    assert storage.saver is None

    # First access initializes saver and creates DB file
    saver1 = await storage.get_or_create_saver()
    assert saver1 is not None
    assert cp_db.is_file()

    # Second access reuses the same connection / saver instance
    saver2 = await storage.get_or_create_saver()
    assert saver1 is saver2

    # Closing resets the saver
    await storage.close()
    assert storage.saver is None


@pytest.mark.anyio
async def test_isolated_checkpoint_state_between_conversations(tmp_path, isolated_storage):
    """Verify distinct conversation IDs maintain isolated checkpoint state in SQLite."""
    cp_db = tmp_path / "isolated_checkpoints.db"
    client = MockCheckpointClient([
        {"message": {"role": "assistant", "content": "Alpha response"}},
        {"message": {"role": "assistant", "content": "Beta response"}},
    ])

    runtime = LangGraphAgentRuntime(
        client=client,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    # Run conversation 1
    res1 = await runtime.run("Alpha query", conversation_id="conv_alpha")
    assert res1.completed is True
    assert res1.final_response == "Alpha response"

    # Run conversation 2
    res2 = await runtime.run("Beta query", conversation_id="conv_beta")
    assert res2.completed is True
    assert res2.final_response == "Beta response"

    # Inspect persisted states
    state_alpha = await runtime.get_state("conv_alpha")
    state_beta = await runtime.get_state("conv_beta")

    assert state_alpha is not None
    assert state_beta is not None
    assert state_alpha.values["prompt"] == "Alpha query"
    assert state_beta.values["prompt"] == "Beta query"
    assert state_alpha.values["final_response"] == "Alpha response"
    assert state_beta.values["final_response"] == "Beta response"

    await runtime.close()


@pytest.mark.anyio
async def test_completed_turn_not_replayed_on_new_turn(tmp_path, isolated_storage):
    """Verify that a completed turn does not replay when a new turn arrives."""
    cp_db = tmp_path / "no_replay.db"
    client = MockCheckpointClient([
        {"message": {"role": "assistant", "content": "Turn 1 answer"}},
        {"message": {"role": "assistant", "content": "Turn 2 answer"}},
    ])

    runtime = LangGraphAgentRuntime(
        client=client,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    cid = "conv_multiturn"
    res1 = await runtime.run("First prompt", conversation_id=cid)
    assert res1.completed is True
    assert res1.rounds_used == 1
    assert await runtime.has_interrupted_turn(cid) is False

    res2 = await runtime.run("Second prompt", conversation_id=cid)
    assert res2.completed is True
    assert res2.prompt == "Second prompt"
    assert res2.final_response == "Turn 2 answer"
    assert res2.rounds_used == 1
    assert await runtime.has_interrupted_turn(cid) is False

    await runtime.close()


@pytest.mark.anyio
async def test_interrupted_turn_resumption_and_crash_recovery(tmp_path, isolated_storage):
    """Verify resumption from interrupted graph execution and cross-process crash recovery."""
    cp_db = tmp_path / "crash_recovery.db"
    client = MockCheckpointClient([
        {"message": {"role": "assistant", "content": "Will be interrupted before finalize"}},
    ])

    cid = "conv_interrupted"

    # Process 1: Runs turn with interrupt_before=['finalize']
    runtime1 = LangGraphAgentRuntime(
        client=client,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        interrupt_before=["finalize"],
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    await runtime1.run("Prompt that gets interrupted", conversation_id=cid)

    # Verify turn is in an interrupted state with 'finalize' pending
    assert await runtime1.has_interrupted_turn(cid) is True
    snapshot1 = await runtime1.get_state(cid)
    assert snapshot1 is not None
    assert snapshot1.next == ("finalize",)
    await runtime1.close()

    # Process 2 (simulating application restart): New runtime pointing to same DB
    runtime2 = LangGraphAgentRuntime(
        client=client,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    # It sees the interrupted turn from disk
    assert await runtime2.has_interrupted_turn(cid) is True

    # Resume the interrupted workflow
    resumed_result = await runtime2.resume(cid)
    assert resumed_result is not None
    assert resumed_result.completed is True
    assert resumed_result.status == "success"
    assert resumed_result.final_response == "Will be interrupted before finalize"

    # Now the turn is fully finished
    assert await runtime2.has_interrupted_turn(cid) is False

    await runtime2.close()


@pytest.mark.anyio
async def test_deduplication_prevents_tool_reexecution_on_resumption(tmp_path, isolated_storage):
    """Verify tool execution deduplication prevents duplicate side-effects upon resumption."""
    cp_db = tmp_path / "dedup_test.db"

    spy_call_count = 0

    async def spy_tool(data: str, **kwargs):
        nonlocal spy_call_count
        spy_call_count += 1
        return {"success": True, "result": f"processed_{data}"}

    registry = create_default_registry()
    registry.register(
        ToolDefinition(
            name="spy_tool",
            description="Tool to monitor execution count",
            parameters_schema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"],
            },
            handler=spy_tool,
        )
    )

    client = MockCheckpointClient([
        # Round 1: Model requests tool call
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_spy_001",
                    "function": {"name": "spy_tool", "arguments": {"data": "test_payload"}},
                }],
            }
        },
        # Round 2: Final response after tool execution
        {"message": {"role": "assistant", "content": "Done processing tool."}},
    ])

    cid = "conv_dedup"

    # Runtime interrupts after executing tools, before finalize
    runtime = LangGraphAgentRuntime(
        client=client,
        registry=registry,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        interrupt_before=["finalize"],
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    # First invocation runs through call_model and execute_tools, then interrupts before finalize
    await runtime.run("Run spy tool", conversation_id=cid)
    assert spy_call_count == 1
    assert await runtime.has_interrupted_turn(cid) is True

    # Resumption: Re-enters the graph from checkpoint and completes
    resumed = await runtime.resume(cid)
    assert resumed is not None
    assert resumed.completed is True
    assert resumed.final_response == "Done processing tool."

    # Crucial check: tool was executed exactly once throughout turn and resumption!
    assert spy_call_count == 1

    # Additional unit check: calling _node_execute_tools with existing tool_records reuses result
    test_state = {
        "current_tool_calls": [{
            "id": "call_spy_001",
            "function": {"name": "spy_tool", "arguments": {"data": "test_payload"}},
        }],
        "task_id": None,
        "conversation_id": cid,
        "rounds_used": 1,
        "tool_records": resumed.tool_records,
        "messages": [],
        "turn_new_messages": [],
    }
    await runtime._node_execute_tools(test_state)
    # Spy tool must STILL NOT be called again
    assert spy_call_count == 1

    await runtime.close()


@pytest.mark.anyio
async def test_action_uncertainty_safeguard_blocks_replay_on_resume(tmp_path, isolated_storage):
    """Verify that an OUTCOME_UNKNOWN action in the ledger blocks automatic replay upon resumption."""
    cp_db = tmp_path / "uncertainty_block.db"

    from storage.action_ledger import normalize_and_hash_arguments
    _, hash_str = normalize_and_hash_arguments({"amount": 100})
    step_name = f"charge_card_action_{hash_str[:12]}"

    # Pre-record an action with OUTCOME_UNKNOWN in the ledger
    task = isolated_storage["task"].create_task(
        task_type="agent_turn", description="Uncertain action task"
    )
    isolated_storage["ledger"].prepare_action(
        task_id=task.id,
        step_name=step_name,
        tool_name="charge_card",
        arguments={"amount": 100},
        initial_status=ActionLedgerStatus.EXECUTING,
    )
    # Transition to OUTCOME_UNKNOWN
    attempt = isolated_storage["ledger"].get_latest_attempt(f"{task.id}:{step_name}")
    isolated_storage["ledger"].update_status(attempt.id, ActionLedgerStatus.OUTCOME_UNKNOWN)

    client = MockCheckpointClient([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_charge_1",
                    "function": {"name": "charge_card", "arguments": {"amount": 100}},
                }],
            }
        }
    ])

    async def mock_charge(amount: int, **kwargs):
        return {"success": True, "charged": amount}

    registry = create_default_registry()
    registry.register(
        ToolDefinition(
            name="charge_card",
            description="Charge a credit card",
            parameters_schema={
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": ["amount"],
            },
            handler=mock_charge,
        )
    )

    runtime = LangGraphAgentRuntime(
        client=client,
        registry=registry,
        checkpoint_db_path=cp_db,
        enable_checkpointing=True,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("Charge $100", conversation_id="conv_uncertain", task_id=task.id)
    assert result.completed is False
    assert result.status == "action_uncertainty_blocked"
    assert "unknown outcome" in result.error_message
    assert "Automatic replay is prohibited" in result.error_message

    await runtime.close()


@pytest.mark.anyio
async def test_checkpointing_can_be_disabled(tmp_path, isolated_storage):
    """Verify that checkpointing can be disabled cleanly and execution proceeds normally."""
    client = MockCheckpointClient([
        {"message": {"role": "assistant", "content": "Stateless response"}},
    ])

    runtime = LangGraphAgentRuntime(
        client=client,
        enable_checkpointing=False,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    assert runtime.enable_checkpointing is False
    assert runtime.checkpointer is None

    result = await runtime.run("Hello without checkpoints", conversation_id="conv_disabled")
    assert result.completed is True
    assert result.final_response == "Stateless response"

    # State inspection methods cleanly return None / False
    assert await runtime.get_state("conv_disabled") is None
    assert await runtime.has_interrupted_turn("conv_disabled") is False
    assert await runtime.resume("conv_disabled") is None

    await runtime.close()


@pytest.mark.anyio
async def test_create_agent_runtime_factory_wiring(tmp_path):
    """Verify create_agent_runtime factory forwards checkpointer settings and switches appropriately."""
    cp_db = tmp_path / "factory_checkpoints.db"

    # 1. Default creation enables checkpointing
    rt1 = create_agent_runtime(checkpoint_db_path=cp_db, enable_checkpointing=True)
    assert isinstance(rt1, LangGraphAgentRuntime)
    assert rt1.enable_checkpointing is True

    # 2. Explicitly disabled checkpointing
    rt2 = create_agent_runtime(enable_checkpointing=False)
    assert isinstance(rt2, LangGraphAgentRuntime)
    assert rt2.enable_checkpointing is False
    assert rt2.checkpointer is None
