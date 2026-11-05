"""Unit tests for JARVIS AgentRuntime and OllamaClient with mocked responses."""
from typing import Any, Dict, List, Optional
import pytest

from agent.ollama_client import (
    LLMClientProtocol,
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from agent.runtime import AgentRuntime
from agent.tool_registry import create_default_registry


class MockOllamaClient(OllamaClient):
    """Test client returning pre-configured sequence of responses without real HTTP calls."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        super().__init__(base_url="http://127.0.0.1:11434")
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
            raise OllamaResponseError("No further mock responses configured.")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return next_resp


@pytest.mark.anyio
async def test_runtime_direct_response_without_tools():
    mock_responses = [{
        "message": {
            "role": "assistant",
            "content": "Good afternoon, sir. How may I be of assistance?",
        }
    }]
    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Hello JARVIS")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 1
    assert result.final_response == "Good afternoon, sir. How may I be of assistance?"
    assert len(result.tool_records) == 0


@pytest.mark.anyio
async def test_runtime_successful_tool_call_flow():
    # Sequence: 1. Model requests tool -> 2. Model receives result and answers
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc123",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "(125 * 8) + 40"},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "The result is 1040.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Calculate (125 * 8) + 40")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 2
    assert result.final_response == "The result is 1040."
    assert len(result.tool_records) == 1

    record = result.tool_records[0]
    assert record.tool_name == "calculate"
    assert record.arguments == {"expression": "(125 * 8) + 40"}
    assert record.result == {"success": True, "result": 1040}

    # Verify that the tool result was transmitted back in the conversation history
    history = client.call_history[1]["messages"]
    tool_msg = next((m for m in history if m.get("role") == "tool"), None)
    assert tool_msg is not None
    assert tool_msg["tool_call_id"] == "call_abc123"
    assert '"success": true' in tool_msg["content"].lower()
    assert "1040" in tool_msg["content"]


@pytest.mark.anyio
async def test_runtime_unknown_tool_rejection_and_graceful_recovery():
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_bad",
                    "function": {
                        "name": "unauthorized_command",
                        "arguments": {},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "I apologize, but that capability is not authorized.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Run unauthorized command")

    assert result.completed is True
    assert result.status == "success"
    assert len(result.tool_records) == 1
    record = result.tool_records[0]
    assert record.result["success"] is False
    assert "is not registered" in record.result["error"]


@pytest.mark.anyio
async def test_runtime_malformed_arguments_handling():
    # Model sends empty arguments to calculate which requires 'expression'
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_missing_arg",
                    "function": {
                        "name": "calculate",
                        "arguments": {},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "Calculation could not be performed due to missing parameters.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Calculate nothing")

    assert result.completed is True
    assert len(result.tool_records) == 1
    assert result.tool_records[0].result["success"] is False
    assert "Invalid arguments" in result.tool_records[0].result["error"]


@pytest.mark.anyio
async def test_runtime_max_rounds_enforcement():
    # Model enters an infinite loop requesting tools
    loop_response = {
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

    mock_responses = [loop_response, loop_response, loop_response]
    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client, max_rounds=2)

    result = await runtime.run("Keep looping")

    assert result.completed is False
    assert result.status == "max_rounds_reached"
    assert result.rounds_used == 2
    assert len(result.tool_records) == 2


@pytest.mark.anyio
async def test_runtime_handles_client_connection_error():
    client = MockOllamaClient([OllamaConnectionError("Connection refused")])
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Test connection failure")

    assert result.completed is False
    assert result.status == "error"
    assert "Connection refused" in (result.error_message or "")


def test_ollama_client_loopback_security_validation():
    # Valid loopback addresses
    c1 = OllamaClient(base_url="http://127.0.0.1:11434")
    assert c1.base_url == "http://127.0.0.1:11434"

    c2 = OllamaClient(base_url="http://localhost:11434/")
    assert c2.base_url == "http://localhost:11434"

    # Non-loopback addresses must be rejected
    with pytest.raises(ValueError, match="Security violation"):
        OllamaClient(base_url="http://192.168.1.100:11434")

    with pytest.raises(ValueError, match="Security violation"):
        OllamaClient(base_url="http://remote-server.com:11434")

    # Non-http scheme must be rejected
    with pytest.raises(ValueError, match="must be 'http'"):
        OllamaClient(base_url="https://localhost:11434")


@pytest.mark.anyio
async def test_runtime_open_website_tool_flow(monkeypatch: pytest.MonkeyPatch):
    import subprocess
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "tools.browser.default_browser_controller._launcher",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, returncode=0),
    )

    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_web_1",
                    "function": {
                        "name": "open_website",
                        "arguments": {"url": "news.ycombinator.com"},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "I have dispatched Hacker News to your browser, sir.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Open Hacker News")

    assert result.completed is True
    assert result.status == "success"
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "open_website"
    assert result.tool_records[0].result["success"] is True
    assert result.tool_records[0].result["result"]["url"] == "https://news.ycombinator.com"
    assert "Hacker News" in result.final_response



@pytest.mark.anyio
async def test_runtime_music_tool_flow(monkeypatch: pytest.MonkeyPatch):
    # Mock default music controller so no real audio plays
    monkeypatch.setattr(
        "tools.music.default_music_controller.play",
        lambda: {"success": True, "status": "playing", "message": "Now playing garage_music.mp3"},
    )

    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_music_1",
                    "function": {
                        "name": "play_music",
                        "arguments": {},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "Playing your music track now, sir.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Play some music")

    assert result.completed is True
    assert result.status == "success"
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "play_music"
    assert result.tool_records[0].result["success"] is True
    assert "Playing your music track" in result.final_response


@pytest.mark.anyio
async def test_runtime_token_limit_truncated_response():
    mock_responses = [
        {
            "done_reason": "length",
            "message": {
                "role": "assistant",
                "content": "",
            },
        }
    ]
    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Explain quantum computing in detail")
    assert result.completed is False
    assert result.status == "truncated"
    assert "token limit" in result.final_response.lower()


@pytest.mark.anyio
async def test_runtime_empty_response_following_tool_detected():
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_calc_1",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "2 + 2"},
                    },
                }],
            }
        },
        {
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "   ",
            },
        },
    ]
    client = MockOllamaClient(mock_responses)
    runtime = AgentRuntime(client=client)

    result = await runtime.run("Calculate 2 + 2")
    assert result.completed is False
    assert result.status == "empty_response"
    assert result.final_response == ""


class PureProtocolMockClient:
    """Mock client that satisfies LLMClientProtocol purely via structural subtyping without inheriting from OllamaClient."""

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
            raise RuntimeError("No further mock responses configured.")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return next_resp


@pytest.mark.anyio
async def test_runtime_with_pure_protocol_mock_client():
    """Verify AgentRuntime accepts a pure mock satisfying LLMClientProtocol without OllamaClient inheritance."""
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_calc_proto",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "100 * 5"},
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "The result of 100 * 5 is 500.",
            }
        },
    ]
    client = PureProtocolMockClient(mock_responses)
    assert isinstance(client, LLMClientProtocol)
    assert not isinstance(client, OllamaClient)

    runtime = AgentRuntime(client=client)
    result = await runtime.run("What is 100 * 5?")

    assert result.completed is True
    assert result.status == "success"
    assert result.rounds_used == 2
    assert "500" in result.final_response
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "calculate"
    assert len(client.call_history) == 2



