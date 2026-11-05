"""Tests for JARVIS Bridge Diagnostics, Telemetry, and Conversation History."""
import pytest
from fastapi.testclient import TestClient

from bridge import app, get_memory_telemetry, get_voice_engine_state
from config.settings import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_memory_telemetry_measurements():
    """Verify memory metrics return genuine non-negative values and safe labels."""
    mem = get_memory_telemetry()
    assert isinstance(mem, dict)
    assert "process_rss_mb" in mem
    assert "system_total_gb" in mem
    assert "display" in mem
    # Process RSS must be a positive float when available
    if mem["process_rss_mb"] is not None:
        assert mem["process_rss_mb"] > 0
    # Display must not be empty
    assert mem["display"] != ""
    assert mem["display"] != "2.1 GB / 8.0 GB"  # Ensure old fake metric is never emitted


def test_voice_engine_state():
    """Verify voice engine reports UNAVAILABLE when not initialized rather than fake READY."""
    state = get_voice_engine_state()
    assert state in ("UNAVAILABLE", "INITIALIZING", "READY")
    # By default in bridge without initialized TTS, it should be UNAVAILABLE
    assert state == "UNAVAILABLE"


def test_diagnostics_endpoint(client):
    """Verify /api/diagnostics returns read-only real runtime telemetry."""
    response = client.get("/api/diagnostics")
    assert response.status_code == 200
    data = response.json()

    # Bridge health & readiness
    assert data["bridge"]["status"] == "READY"
    assert data["bridge"]["port"] == 8000
    assert data["bridge"]["uptime_seconds"] >= 0

    # Model and Provider
    assert data["router"]["provider"] == "Ollama"
    assert data["router"]["model"] == settings.llm_model
    assert settings.llm_model.upper() in data["router"]["display"]

    # Voice Engine State
    assert data["voice_engine"]["state"] in ("UNAVAILABLE", "INITIALIZING", "READY")

    # Latency: before any chat, latency should have display N/A or a real float
    assert "latency" in data
    assert "memory" in data
    assert data["memory"]["display"] != ""


def test_status_endpoint_no_private_paths(client):
    """Verify /api/status does not expose private absolute filesystem paths."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert data["llm_model"] == settings.llm_model
    assert "voice_engine_state" in data
    assert "memory" in data

    # Verify no private absolute paths are leaked in database info
    db_info = data.get("database", {})
    assert "path" not in db_info
    assert "name" in db_info
    assert db_info["name"] == settings.database_path.name


def test_chat_and_latency_measurement(client):
    """Verify /api/chat measures real latency and stores genuine conversation history."""
    conv_id = "test_telemetry_session"
    req_payload = {
        "message": "calculate 25 * 4",
        "conversation_id": conv_id,
    }
    resp = client.post("/api/chat", json=req_payload)
    assert resp.status_code == 200
    chat_data = resp.json()

    assert chat_data["status"] == "success"
    assert chat_data["conversation_id"] == conv_id
    assert chat_data["latency_ms"] > 0
    assert "100" in chat_data["response"]

    # Now verify diagnostics reports the updated latest chat latency
    diag_resp = client.get("/api/diagnostics")
    diag_data = diag_resp.json()
    assert diag_data["latency"]["latest_ms"] is not None
    assert diag_data["latency"]["latest_ms"] > 0
    assert diag_data["latency"]["display"].endswith("ms")

    # Verify /api/history returns genuine persisted messages
    hist_resp = client.get(f"/api/history?conversation_id={conv_id}")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) >= 2
    assert history[0]["role"] == "user"
    assert "calculate 25 * 4" in history[0]["content"]
    assert history[1]["role"] == "assistant"
    assert "100" in history[1]["content"]
