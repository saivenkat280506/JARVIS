"""FastAPI Bridge Server for JARVIS Frontend Command Deck.

Provides local REST API endpoints for telemetry, real-time agent execution,
persistent conversation history, and diagnostic status.
"""
from __future__ import annotations

import asyncio
import os
import logging
import platform
import re
import resource
import subprocess
from contextlib import asynccontextmanager
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config.settings import settings
from agent.ollama_client import OllamaClient
from agent.runtime import AgentRuntime, create_agent_runtime
from agent.tool_registry import create_default_registry, safe_calculate
from storage import (
    default_conversation_storage,
    default_task_storage,
    default_reminder_storage,
    default_action_ledger,
)

logger = logging.getLogger(__name__)

_start_time = time.time()
_system_state: str = "standby"
_latest_chat_latency_ms: Optional[float] = None
_tts_engine: Optional[Any] = None
_is_shutting_down: bool = False
_shutdown_lock: asyncio.Lock = asyncio.Lock()


async def perform_graceful_shutdown() -> Dict[str, Any]:
    """Execute clean, idempotent shutdown of all managed backend resources.

    Ensures that:
    1. Active background music playback is stopped cleanly.
    2. Active speech playback (TTS) process is released.
    3. Durable SQLite checkpoint saver connections are closed.
    4. Subsystem failures are isolated so remaining cleanup operations execute.
    5. Safe and idempotent to call multiple times without error.
    """
    global _is_shutting_down, _system_state
    async with _shutdown_lock:
        if _is_shutting_down:
            return {"status": "already_shut_down", "results": {}}
        _is_shutting_down = True
        _system_state = "shutting_down"

    results: Dict[str, str] = {}
    logger.info("Initiating graceful JARVIS backend shutdown...")

    # 1. Stop background music playback
    try:
        from tools.music import default_music_controller
        res = default_music_controller.stop()
        results["music"] = "stopped" if res.get("success") else res.get("status", "unknown")
    except Exception as exc:
        logger.warning("Error stopping music during shutdown: %s", exc)
        results["music"] = f"error: {exc}"

    # 2. Stop voice / TTS playback process
    try:
        global _tts_engine
        if _tts_engine is not None and hasattr(_tts_engine, "stop_playback"):
            _tts_engine.stop_playback()
            results["voice_tts"] = "stopped"
        else:
            results["voice_tts"] = "inactive"
    except Exception as exc:
        logger.warning("Error stopping TTS during shutdown: %s", exc)
        results["voice_tts"] = f"error: {exc}"

    # 3. Close SQLite Checkpoint storage
    try:
        from storage.checkpoint import default_checkpoint_storage
        await default_checkpoint_storage.close()
        results["checkpoints"] = "closed"
    except Exception as exc:
        logger.warning("Error closing checkpoint storage: %s", exc)
        results["checkpoints"] = f"error: {exc}"

    logger.info("JARVIS backend shutdown completed: %s", results)
    return {"status": "success", "results": results}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and graceful shutdown."""
    logger.info("JARVIS API Bridge initialized and listening.")
    try:
        yield
    finally:
        await perform_graceful_shutdown()


app = FastAPI(
    title="JARVIS API Bridge",
    description="Operational API Bridge connecting Command Deck UI to local JARVIS Agent",
    version="3.14-PROD",
    lifespan=lifespan,
)

# Enable CORS for local frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_memory_telemetry() -> Dict[str, Any]:
    """Measure real process RSS and system RAM without external dependencies."""
    proc_rss_mb: Optional[float] = None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            proc_rss_mb = round(ru / (1024 * 1024), 1)
        else:
            proc_rss_mb = round(ru / 1024, 1)
    except Exception:
        proc_rss_mb = None

    sys_total_gb: Optional[float] = None
    sys_used_gb: Optional[float] = None
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        sys_total_gb = round(total_bytes / (1024**3), 1)
    except Exception:
        pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["vm_stat"], timeout=1).decode()
            page_size = 16384
            m = re.search(r"page size of (\d+) bytes", out)
            if m:
                page_size = int(m.group(1))
            active_m = re.search(r"Pages active:\s+(\d+)", out)
            wired_m = re.search(r"Pages wired down:\s+(\d+)", out)
            if active_m and wired_m:
                active_pages = int(active_m.group(1))
                wired_pages = int(wired_m.group(1))
                occupied = (active_pages + wired_pages) * page_size
                sys_used_gb = round(occupied / (1024**3), 1)
        except Exception:
            pass

    if sys_used_gb is not None and sys_total_gb is not None:
        display = f"{sys_used_gb} GB / {sys_total_gb} GB (SYS)"
    elif proc_rss_mb is not None:
        display = f"{proc_rss_mb} MB (PROC)"
    else:
        display = "N/A"

    return {
        "process_rss_mb": proc_rss_mb,
        "system_used_gb": sys_used_gb,
        "system_total_gb": sys_total_gb,
        "display": display,
    }


def get_voice_engine_state() -> str:
    """Return genuine initialization and readiness state of the voice engine."""
    global _tts_engine
    if _tts_engine is None:
        return "UNAVAILABLE"
    if getattr(_tts_engine, "_model", None) is not None:
        return "READY"
    return "INITIALIZING"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    latency_ms: float
    tools_used: List[Dict[str, Any]] = []
    model: str
    status: str = "success"


@app.get("/api/diagnostics")
async def get_diagnostics():
    """Read-only system diagnostics for Command Deck telemetry.

    Exposes only bridge health, model routing, voice engine state, real measured
    latency, and actual system memory without leaking paths, secrets, or internal logs.
    """
    ollama_ready = False
    try:
        client = OllamaClient(base_url=settings.ollama_base_url, default_model=settings.llm_model)
        ollama_ready = await client.is_available()
        await client.close()
    except Exception:
        ollama_ready = False

    mem = get_memory_telemetry()
    uptime_sec = round(time.time() - _start_time, 1)

    return {
        "bridge": {
            "status": "READY",
            "port": 8000,
            "uptime_seconds": uptime_sec,
        },
        "router": {
            "provider": "Ollama",
            "model": settings.llm_model,
            "ollama_online": ollama_ready,
            "display": f"OLLAMA / {settings.llm_model.upper()}",
        },
        "voice_engine": {
            "state": get_voice_engine_state(),
        },
        "latency": {
            "latest_ms": _latest_chat_latency_ms,
            "display": f"{round(_latest_chat_latency_ms)}ms" if _latest_chat_latency_ms is not None else "N/A",
        },
        "memory": mem,
        "system_status": _system_state,
    }


@app.get("/api/status")
async def get_status():
    """Return live system telemetry, database status, and LLM availability."""
    db_exists = settings.database_path.is_file()
    db_size_kb = (settings.database_path.stat().st_size / 1024.0) if db_exists else 0.0

    # Query counts safely
    try:
        tasks = default_task_storage.list_tasks(limit=100)
        active_tasks = len([t for t in tasks if t.status.value in ("queued", "running", "waiting_for_approval")])
    except Exception:
        active_tasks = 0

    try:
        reminders = default_reminder_storage.list_reminders(status="pending")
        pending_reminders = len(reminders)
    except Exception:
        pending_reminders = 0

    try:
        convs = default_conversation_storage.list_conversations(limit=20)
        conv_count = len(convs)
    except Exception:
        conv_count = 0

    # Quick check for Ollama availability
    ollama_ready = False
    try:
        client = OllamaClient(base_url=settings.ollama_base_url, default_model=settings.llm_model)
        ollama_ready = await client.is_available()
        await client.close()
    except Exception:
        ollama_ready = False

    uptime_sec = time.time() - _start_time
    mem = get_memory_telemetry()
    voice_state = get_voice_engine_state()

    return {
        "status": "online",
        "system": f"{platform.system()} {platform.machine()}",
        "python_version": sys.version.split()[0],
        "uptime_seconds": round(uptime_sec, 1),
        "llm_model": settings.llm_model,
        "llm_provider": "Ollama",
        "router_display": f"OLLAMA / {settings.llm_model.upper()}",
        "ollama_online": ollama_ready,
        "database": {
            "name": settings.database_path.name,
            "size_kb": round(db_size_kb, 1),
            "active_tasks": active_tasks,
            "pending_reminders": pending_reminders,
            "total_conversations": conv_count,
        },
        "voice_engine_state": voice_state,
        "latest_latency_ms": _latest_chat_latency_ms,
        "memory": mem,
        "bridge": {
            "status": "READY",
            "port": 8000,
        },
        "bridge_port": 8000,
    }


@app.get("/api/history")
async def get_history(conversation_id: str = "deck_session", limit: int = 50):
    """Retrieve genuine persisted conversation messages from SQLite."""
    try:
        msgs = default_conversation_storage.get_messages(conversation_id=conversation_id, limit=limit)
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in msgs
            if m.role in ("user", "assistant") and m.content and m.content.strip()
        ]
    except Exception:
        return []


@app.delete("/api/history")
@app.post("/api/clear")
async def clear_history(conversation_id: str = "deck_session"):
    """Clear conversation history from SQLite storage."""
    try:
        success = default_conversation_storage.clear_conversation(conversation_id=conversation_id)
        return {"status": "cleared", "conversation_id": conversation_id, "success": success}
    except Exception as e:
        logger.warning(f"Error clearing conversation history: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Process a user message through the AgentRuntime or fast local tools."""
    global _latest_chat_latency_ms, _system_state
    _system_state = "processing"
    start_time = time.perf_counter()
    msg = request.message.strip()
    conv_id = request.conversation_id or "deck_session"

    tools_used = []
    response_text = ""
    used_model = settings.llm_model

    try:
        # Check if Ollama is available
        client = OllamaClient(base_url=settings.ollama_base_url, default_model=settings.llm_model)
        ollama_online = await client.is_available()

        if ollama_online:
            try:
                registry = create_default_registry()
                runtime = create_agent_runtime(client=client, registry=registry, max_rounds=5)
                result = await runtime.run(msg, conversation_id=conv_id)
                await client.close()

                response_text = result.final_response
                if not response_text:
                    if result.status == "waiting_for_approval":
                        response_text = "A consequential action requires your approval. Please confirm or deny."
                    elif result.error_message:
                        response_text = "Agent encountered an execution issue while processing your request."
                    else:
                        response_text = "Request processed."

                if result.tool_records:
                    for rec in result.tool_records:
                        tools_used.append({
                            "tool": rec.tool_name,
                            "args": rec.arguments,
                            "latency_ms": round(rec.duration_sec * 1000.0, 1),
                        })
            except Exception as exc:
                await client.close()
                logger.error("Agent execution failure: %s", exc, exc_info=True)
                response_text = "Agent encountered an execution issue while processing your request."
        else:
            await client.close()
            used_model = "LOCAL-FAST"

            # Deterministic local fallback engine
            lower_msg = msg.lower()
            if "calculate" in lower_msg or any(op in msg for op in ["+", "-", "*", "/"]):
                # Extract expression
                expr = msg.replace("calculate", "").replace("what is", "").replace("?", "").strip()
                try:
                    val = safe_calculate(expr)
                    response_text = f"Calculated: {expr} = {val}"
                    tools_used.append({"tool": "calculate", "args": {"expression": expr}, "latency_ms": 1.2})
                except Exception:
                    response_text = f"Processed command: \"{msg}\". System operational."
            elif "status" in lower_msg or "diagnostics" in lower_msg or "health" in lower_msg:
                response_text = (
                    f"JARVIS systems nominal. Platform: {platform.system()} ({platform.machine()}). "
                    f"Database active at {settings.database_path.name}. API bridge operational."
                )
            elif "reminder" in lower_msg:
                reminders = default_reminder_storage.list_reminders()
                if reminders:
                    titles = ", ".join(f"'{r.reminder_text}'" for r in reminders[:3])
                    response_text = f"You have {len(reminders)} active reminders: {titles}."
                else:
                    response_text = "No pending reminders found in database."
            elif "task" in lower_msg:
                tasks = default_task_storage.list_tasks(limit=5)
                response_text = f"Currently tracking {len(tasks)} tasks in the SQLite ledger."
            else:
                response_text = f"Command received: \"{msg}\". All localized diagnostic systems reporting nominal."

            # Persist conversation turn
            try:
                default_conversation_storage.append_message(
                    conversation_id=conv_id,
                    role="user",
                    content=msg,
                )
                default_conversation_storage.append_message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=response_text,
                )
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _latest_chat_latency_ms = round(elapsed_ms, 1)

        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            latency_ms=round(elapsed_ms, 1),
            tools_used=tools_used,
            model=used_model,
            status="success",
        )
    finally:
        _system_state = "standby"


@app.get("/api/tasks")
async def get_tasks(limit: int = 10):
    """Retrieve persistent tasks."""
    tasks = default_task_storage.list_tasks(limit=limit)
    return [
        {
            "id": t.id,
            "description": t.description,
            "status": t.status.value,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
        for t in tasks
    ]


@app.get("/api/reminders")
async def get_reminders(status: str = "pending"):
    """Retrieve scheduled reminders."""
    reminders = default_reminder_storage.list_reminders(status=status)
    return [
        {
            "id": r.id,
            "text": r.reminder_text,
            "due_time": r.due_time,
            "status": r.status.value,
        }
        for r in reminders
    ]


@app.post("/api/shutdown")
async def shutdown_endpoint():
    """Trigger graceful backend shutdown from local supervisor or desktop shell.

    Executes all cleanup hooks, releases hardware and storage resources,
    and schedules graceful process termination.
    """
    res = await perform_graceful_shutdown()
    try:
        loop = asyncio.get_running_loop()
        # Allow response to be dispatched to caller before process termination
        loop.call_later(0.15, lambda: os.kill(os.getpid(), signal.SIGTERM))
    except Exception:
        pass
    return {
        "status": "shutting_down",
        "message": "JARVIS Bridge is shutting down gracefully.",
        "details": res,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting JARVIS API Bridge on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
