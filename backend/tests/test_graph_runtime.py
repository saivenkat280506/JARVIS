"""Focused unit and integration tests for LangGraphAgentRuntime."""
from typing import Any, Dict, List, Optional
import pytest

from agent.graph_runtime import LangGraphAgentRuntime
from agent.ollama_client import (
    LLMClientProtocol,
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from agent.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
)
from storage.conversation import ConversationStorage
from storage.tasks import TaskStatus, TaskStorage
from storage.action_ledger import ActionLedger, ActionLedgerStatus
from security.approval import ActionApprovalManager, ApprovalStatus


class MockGraphClient:
    """Mock LLM client fulfilling LLMClientProtocol for LangGraph runtime tests."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.call_history: List[Dict[str, Any]] = []

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
            raise OllamaResponseError("No further mock responses configured in test.")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return next_resp


@pytest.fixture
def isolated_storage(tmp_path):
    """Provide isolated SQLite storage backends for testing."""
    db_path = tmp_path / "test_graph.db"
    conv_storage = ConversationStorage(db_path=db_path)
    task_storage = TaskStorage(db_path=db_path)
    action_ledger = ActionLedger(db_path=db_path)
    approval_manager = ActionApprovalManager(db_path=db_path)
    return {
        "conv": conv_storage,
        "task": task_storage,
        "ledger": action_ledger,
        "approval": approval_manager,
    }


@pytest.mark.anyio
async def test_graph_initialization_with_mock_client():
    """Verify LangGraphAgentRuntime initializes cleanly with compiled graph and mock client."""
    client = MockGraphClient([])
    assert isinstance(client, LLMClientProtocol)
    assert not isinstance(client, OllamaClient)

    runtime = LangGraphAgentRuntime(client=client)
    assert runtime.graph is not None
    assert runtime.max_rounds == 5


@pytest.mark.anyio
async def test_graph_normal_request_without_tools(isolated_storage):
    """Verify normal conversational request returning final answer without tool calls."""
    mock_responses = [{
        "message": {
            "role": "assistant",
            "content": "Good morning, sir. All systems are operational.",
        }
    }]
    client = MockGraphClient(mock_responses)
    runtime = LangGraphAgentRuntime(
        client=client,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("Status report")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 1
    assert result.final_response == "Good morning, sir. All systems are operational."
    assert len(result.tool_records) == 0
    assert len(client.call_history) == 1


@pytest.mark.anyio
async def test_graph_single_tool_round(isolated_storage):
    """Verify a tool call followed by a model final response."""
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_calc_1",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "25 * 4"},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "The result is 100.",
            }
        },
    ]
    client = MockGraphClient(mock_responses)
    runtime = LangGraphAgentRuntime(
        client=client,
        registry=create_default_registry(),
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("What is 25 times 4?")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 2
    assert "100" in result.final_response
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "calculate"
    assert result.tool_records[0].result["result"] == 100
    assert len(client.call_history) == 2


@pytest.mark.anyio
async def test_graph_multiple_bounded_tool_rounds(isolated_storage):
    """Verify multiple distinct tool rounds executing within bounds."""
    mock_responses = [
        # Round 1: Call get_local_time
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_time_1",
                    "function": {"name": "get_local_time", "arguments": {}},
                }],
            }
        },
        # Round 2: Call calculate
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_calc_2",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "50 + 50"},
                    },
                }],
            }
        },
        # Round 3: Final response
        {
            "message": {
                "role": "assistant",
                "content": "The time was retrieved and 50 plus 50 equals 100.",
            }
        },
    ]
    client = MockGraphClient(mock_responses)
    runtime = LangGraphAgentRuntime(
        client=client,
        registry=create_default_registry(),
        max_rounds=5,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("Get time and calculate 50 + 50")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 3
    assert len(result.tool_records) == 2
    assert result.tool_records[0].tool_name == "get_local_time"
    assert result.tool_records[1].tool_name == "calculate"
    assert len(client.call_history) == 3


@pytest.mark.anyio
async def test_graph_execution_limit_termination(isolated_storage):
    """Verify safe termination when execution limit (max_rounds) is reached."""
    # Loop continuously requesting calculate
    infinite_tool_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_loop",
                "function": {
                    "name": "calculate",
                    "arguments": {"expression": "1 + 1"},
                },
            }],
        }
    }
    mock_responses = [infinite_tool_response, infinite_tool_response, infinite_tool_response]
    client = MockGraphClient(mock_responses)
    runtime = LangGraphAgentRuntime(
        client=client,
        registry=create_default_registry(),
        max_rounds=2,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("Loop forever")

    assert result.completed is False
    assert result.status == "max_rounds_reached"
    assert "Maximum tool execution rounds reached" in result.final_response
    assert result.rounds_used >= 2


@pytest.mark.anyio
async def test_graph_tool_failure_handled_safely(isolated_storage):
    """Verify tool execution failures are captured and surfaced cleanly."""
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_div_zero",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "10 / 0"},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "Division by zero is undefined.",
            }
        },
    ]
    client = MockGraphClient(mock_responses)
    runtime = LangGraphAgentRuntime(
        client=client,
        registry=create_default_registry(),
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result = await runtime.run("Divide 10 by zero")

    assert result.completed is True
    assert result.status == "success"
    assert len(result.tool_records) == 1
    assert result.tool_records[0].result["success"] is False
    assert "Division by zero" in result.tool_records[0].result["error"]
    assert "Division by zero is undefined." in result.final_response


@pytest.mark.anyio
async def test_graph_approval_requirements_enforced(isolated_storage):
    """Verify that consequential action approval requirements remain strictly enforced in the graph."""
    registry = create_default_registry()

    # Register a consequential test tool
    async def mock_sensitive_action(target: str, **kwargs):
        if not kwargs.get("approval_token"):
            action_rec = isolated_storage["approval"].create_pending_action(
                tool_name="sensitive_action",
                action_type="delete_data",
                target=target,
                arguments={"target": target},
                description=f"Delete all data in {target}",
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": action_rec.action_id,
                "consequential_reason": f"Delete all data in {target}",
            }
        return {"success": True, "message": f"Successfully deleted {target}"}

    registry.register(
        ToolDefinition(
            name="sensitive_action",
            description="Perform a sensitive deletion",
            parameters_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            handler=mock_sensitive_action,
        )
    )

    # 1. Turn 1: Model requests sensitive action
    client1 = MockGraphClient([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_sens_1",
                    "function": {
                        "name": "sensitive_action",
                        "arguments": {"target": "user_records"},
                    },
                }],
            }
        }
    ])

    runtime = LangGraphAgentRuntime(
        client=client1,
        registry=registry,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result1 = await runtime.run("Delete user records", conversation_id="conv_hitl_test")

    # Action should NOT have executed; must be awaiting confirmation
    assert result1.completed is False
    assert result1.status == "waiting_for_approval"

    pending = isolated_storage["approval"].get_latest_pending()
    assert pending is not None
    assert pending.tool_name == "sensitive_action"
    assert pending.status == ApprovalStatus.PENDING

    # 2. Turn 2: User confirms with 'yes proceed'
    client2 = MockGraphClient([])
    runtime_confirm = LangGraphAgentRuntime(
        client=client2,
        registry=registry,
        storage=isolated_storage["conv"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        approval_manager=isolated_storage["approval"],
    )

    result2 = await runtime_confirm.run("yes proceed", conversation_id="conv_hitl_test")

    assert result2.completed is True
    assert result2.status == "success"
    assert "Confirmed and executed" in result2.final_response
    assert "Successfully deleted user_records" in result2.final_response

    # Verify action ledger status updated to SUCCEEDED
    att = isolated_storage["ledger"].get_attempt_by_action_id(pending.action_id)
    assert att is not None
    assert att.status == ActionLedgerStatus.SUCCEEDED
