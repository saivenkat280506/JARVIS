"""Integration tests for production LangGraph agent runtime.

Verifies:
1. Production runtime factory correctly instantiates LangGraphAgentRuntime.
2. Shared singleton instances (storage, approval, ledger, tasks) are maintained.
3. Bridge /api/chat route uses LangGraphAgentRuntime and preserves API contracts.
4. Bridge /api/chat handles tool execution via LangGraphAgentRuntime.
5. Bridge /api/chat preserves consequential action approval boundaries.
6. Bridge /api/chat sanitizes runtime exceptions without leaking secrets or tracebacks.
7. VoicePipeline integrates with LangGraphAgentRuntime produced by the factory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.graph_runtime import LangGraphAgentRuntime
from agent.ollama_client import LLMClientProtocol, OllamaResponseError
from agent.runtime import AgentRunResult, AgentRuntime, create_agent_runtime
from agent.tool_registry import ToolDefinition, ToolRegistry, create_default_registry
from bridge import app
from config.settings import settings
from security.approval import ActionApprovalManager, ApprovalStatus, default_approval_manager
from storage.action_ledger import ActionLedger, ActionLedgerStatus, default_action_ledger
from storage.conversation import ConversationStorage, default_conversation_storage
from storage.tasks import TaskRecord, TaskStatus, TaskStorage, default_task_storage
from voice.pipeline import VoicePipeline
from voice.stt import AudioRecorder, SpeechToTextEngine, STTResult
from voice.tts import TextToSpeechEngine, TTSResult


class MockGraphLLMClient:
    """Mock LLM client fulfilling LLMClientProtocol for production runtime tests."""

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

    async def is_available(self) -> bool:
        return True

    async def close(self) -> None:
        pass


@pytest.fixture
def isolated_storage(tmp_path: Path):
    """Provide isolated SQLite storage backends for testing."""
    db_path = tmp_path / "test_prod.db"
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


def test_production_factory_selects_langgraph_runtime():
    """Verify that create_agent_runtime returns an instance of LangGraphAgentRuntime."""
    runtime = create_agent_runtime()
    assert isinstance(runtime, LangGraphAgentRuntime)
    assert hasattr(runtime, "graph")
    assert runtime.graph is not None
    # Reuses production singletons by default
    assert runtime.storage == default_conversation_storage
    assert runtime.approval_manager == default_approval_manager
    assert runtime.task_storage == default_task_storage
    assert runtime.action_ledger == default_action_ledger
    assert runtime.max_rounds == 5


def test_production_factory_accepts_dependency_injection(isolated_storage):
    """Verify that create_agent_runtime allows explicit DI overrides."""
    mock_client = MockGraphLLMClient([])
    custom_registry = ToolRegistry()
    runtime = create_agent_runtime(
        client=mock_client,
        registry=custom_registry,
        storage=isolated_storage["conv"],
        approval_manager=isolated_storage["approval"],
        task_storage=isolated_storage["task"],
        action_ledger=isolated_storage["ledger"],
        max_rounds=3,
        model="custom-prod-model",
    )
    assert isinstance(runtime, LangGraphAgentRuntime)
    assert runtime.client is mock_client
    assert runtime.registry is custom_registry
    assert runtime.storage is isolated_storage["conv"]
    assert runtime.approval_manager is isolated_storage["approval"]
    assert runtime.task_storage is isolated_storage["task"]
    assert runtime.action_ledger is isolated_storage["ledger"]
    assert runtime.max_rounds == 3
    assert runtime.model == "custom-prod-model"


def test_bridge_chat_uses_langgraph_runtime_direct_answer(isolated_storage, monkeypatch):
    """Verify /api/chat executes through LangGraphAgentRuntime and maintains API contract."""
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "Good morning. All systems are fully operational.",
            }
        }
    ]
    mock_client = MockGraphLLMClient(mock_responses)

    # Patch OllamaClient in bridge to use our mock
    monkeypatch.setattr("bridge.OllamaClient", lambda **kwargs: mock_client)

    client = TestClient(app)
    conv_id = "test_bridge_prod_conv"
    payload = {"message": "System status check", "conversation_id": conv_id}

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["conversation_id"] == conv_id
    assert data["response"] == "Good morning. All systems are fully operational."
    assert data["tools_used"] == []
    assert data["latency_ms"] > 0
    assert data["model"] == settings.llm_model


def test_bridge_chat_uses_langgraph_runtime_tool_execution(isolated_storage, monkeypatch):
    """Verify /api/chat executes tools through LangGraphAgentRuntime and preserves tool telemetry."""
    mock_responses = [
        # Round 1: Model invokes calculate
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_calc_99",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "21 * 4"},
                    },
                }],
            }
        },
        # Round 2: Model provides final response
        {
            "message": {
                "role": "assistant",
                "content": "The result of 21 multiplied by 4 is 84.",
            }
        },
    ]
    mock_client = MockGraphLLMClient(mock_responses)
    monkeypatch.setattr("bridge.OllamaClient", lambda **kwargs: mock_client)

    client = TestClient(app)
    conv_id = "test_bridge_tool_conv"
    payload = {"message": "Multiply 21 by 4", "conversation_id": conv_id}

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "84" in data["response"]
    assert len(data["tools_used"]) == 1
    assert data["tools_used"][0]["tool"] == "calculate"
    assert data["tools_used"][0]["args"] == {"expression": "21 * 4"}
    assert data["tools_used"][0]["latency_ms"] >= 0


def test_bridge_chat_runtime_exception_does_not_leak_secrets(monkeypatch):
    """Verify runtime exceptions do not expose connection strings, passwords, or internal tracebacks."""
    class FailingClient:
        async def is_available(self) -> bool:
            return True
        async def close(self) -> None:
            pass

    async def mock_run(*args, **kwargs):
        raise RuntimeError("DB_SECRET=postgres://admin:super_secret_token_xyz99@10.0.0.1:5432/core connection failed")

    monkeypatch.setattr("bridge.OllamaClient", lambda **kwargs: FailingClient())
    monkeypatch.setattr("bridge.create_agent_runtime", lambda **kwargs: MagicMock(run=mock_run))

    client = TestClient(app)
    payload = {"message": "trigger failure", "conversation_id": "test_err_conv"}

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Public response must not leak the secret token, connection URI, or traceback
    assert "super_secret_token_xyz99" not in data["response"]
    assert "postgres://" not in data["response"]
    assert "Traceback" not in data["response"]
    assert "Agent encountered an execution issue while processing your request." in data["response"]


@pytest.mark.anyio
async def test_voice_pipeline_production_runtime_integration():
    """Verify VoicePipeline operates seamlessly with create_agent_runtime."""
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "Voice turn successfully completed via LangGraph runtime.",
            }
        }
    ]
    mock_client = MockGraphLLMClient(mock_responses)
    runtime = create_agent_runtime(client=mock_client)
    assert isinstance(runtime, LangGraphAgentRuntime)

    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.is_input_available.return_value = True
    mock_recorder.sample_rate = 16000
    dummy_audio = np.ones(16000, dtype=np.float32)
    mock_recorder.record_until_silence.return_value = dummy_audio

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio.return_value = STTResult(
        text="Status report", latency_sec=0.05, audio_duration_sec=1.0, is_empty=False
    )

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.speak.return_value = TTSResult(
        text="Voice turn successfully completed via LangGraph runtime.",
        latency_sec=0.05,
        audio_duration_sec=1.0,
        played=True,
    )

    pipeline = VoicePipeline(
        runtime=runtime,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        recorder=mock_recorder,
        conversation_id="voice_prod_test",
    )

    result = await pipeline.execute_turn_from_audio(dummy_audio, sample_rate=16000)
    assert result.success is True
    assert "Voice turn successfully completed" in result.final_response
    assert result.metrics is not None
    assert result.metrics.agent_runtime_ms is not None
    assert result.metrics.agent_runtime_ms > 0


def test_bridge_chat_consequential_approval_two_turn_flow(monkeypatch, isolated_storage):
    """Verify /api/chat enforces approval on Turn 1 and executes on Turn 2 confirmation."""
    async def mock_transfer_funds(amount: int, **kwargs):
        if not kwargs.get("approval_token"):
            act = isolated_storage["approval"].create_pending_action(
                tool_name="transfer_funds",
                action_type="transfer",
                target=f"amount_{amount}",
                arguments={"amount": amount},
                description=f"Transfer ${amount}",
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": act.action_id,
                "consequential_reason": f"Transfer ${amount}",
            }
        return {"success": True, "message": f"Successfully transferred ${amount}"}

    test_registry = create_default_registry()
    test_registry.register(
        ToolDefinition(
            name="transfer_funds",
            description="Transfer funds",
            parameters_schema={
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": ["amount"],
            },
            handler=mock_transfer_funds,
        )
    )

    # Turn 1: Model calls transfer_funds
    turn1_client = MockGraphLLMClient([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_trans_1",
                    "function": {
                        "name": "transfer_funds",
                        "arguments": {"amount": 500},
                    },
                }],
            }
        }
    ])

    # Turn 2: User confirms, model doesn't need to be called
    turn2_client = MockGraphLLMClient([])

    # Patch create_default_registry in bridge to use test_registry
    monkeypatch.setattr("bridge.create_default_registry", lambda: test_registry)
    # Patch storage components
    monkeypatch.setattr("agent.runtime.default_conversation_storage", isolated_storage["conv"])
    monkeypatch.setattr("agent.runtime.default_approval_manager", isolated_storage["approval"])
    monkeypatch.setattr("agent.runtime.default_task_storage", isolated_storage["task"])
    monkeypatch.setattr("agent.runtime.default_action_ledger", isolated_storage["ledger"])
    monkeypatch.setattr("agent.graph_runtime.default_conversation_storage", isolated_storage["conv"])
    monkeypatch.setattr("agent.graph_runtime.default_approval_manager", isolated_storage["approval"])
    monkeypatch.setattr("agent.graph_runtime.default_task_storage", isolated_storage["task"])
    monkeypatch.setattr("agent.graph_runtime.default_action_ledger", isolated_storage["ledger"])

    conv_id = "test_bridge_hitl_conv"
    client = TestClient(app)

    # 1. Turn 1
    monkeypatch.setattr("bridge.OllamaClient", lambda **kwargs: turn1_client)
    resp1 = client.post("/api/chat", json={"message": "Send $500", "conversation_id": conv_id})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "explicit approval" in data1["response"] or "approval" in data1["response"].lower()

    pending = isolated_storage["approval"].get_latest_pending()
    assert pending is not None
    assert pending.tool_name == "transfer_funds"
    assert pending.status == ApprovalStatus.PENDING

    # 2. Turn 2
    monkeypatch.setattr("bridge.OllamaClient", lambda **kwargs: turn2_client)
    resp2 = client.post("/api/chat", json={"message": "confirm and proceed", "conversation_id": conv_id})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "Confirmed and executed" in data2["response"]
    assert "Successfully transferred $500" in data2["response"]

    # Verify ledger status
    att = isolated_storage["ledger"].get_attempt_by_action_id(pending.action_id)
    assert att is not None
    assert att.status == ActionLedgerStatus.SUCCEEDED

