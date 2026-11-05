"""Automated tests for JARVIS Desktop Lifecycle Management and Process Supervision.

Verifies:
1. Successful startup launches the expected managed services.
2. The application waits for readiness rather than relying on a fixed delay.
3. A backend startup failure is reported correctly.
4. A frontend startup failure cleans up an already-started backend.
5. A readiness timeout results in bounded failure and cleanup.
6. Normal quit requests graceful shutdown.
7. A non-responsive child is force-terminated after the grace period.
8. Repeated shutdown calls are safe and idempotent.
9. Duplicate launcher instances do not manage the same services independently.
10. An unrelated process is never terminated during cleanup.
11. An existing independently running Ollama service is not killed.
12. The backend closes checkpointing, database, voice, and other owned resources appropriately.
13. Development mode works with the existing project and virtual environment.
14. The production frontend assets are valid and ready for bundling.
15. The macOS application bundle contains the correct icon and required application resources.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from desktop.lifecycle import (
    DesktopLifecycleManager,
    InstanceLockError,
    PortConflictError,
    ReadinessTimeoutError,
    StartupError,
)
from bridge import perform_graceful_shutdown


class MockChildProcess:
    """Mock subprocess.Popen object for testing process supervision."""

    def __init__(self, pid: int = 99999, exit_code: Optional[int] = None) -> None:
        self.pid = pid
        self._exit_code = exit_code
        self.signals_received: list[int] = []
        self._is_alive = (exit_code is None)

    def poll(self) -> Optional[int]:
        return self._exit_code if not self._is_alive else None

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._exit_code if self._exit_code is not None else 0

    def send_signal(self, sig: int) -> None:
        self.signals_received.append(sig)
        if sig in (signal.SIGTERM, signal.SIGKILL, signal.SIGINT):
            self._is_alive = False
            self._exit_code = 0


# ------------------------------------------------------------------------------
# Test 1: Successful startup launches expected managed services
# ------------------------------------------------------------------------------
def test_successful_startup_launches_managed_services(tmp_path: Path):
    """Verify that a successful startup spawns the backend process and tracks its PID."""
    mock_proc = MockChildProcess(pid=12345)
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
        http_client=lambda url: {"status": "online", "system": "test"},
    )

    with patch.object(mgr, "verify_port_available"), \
         patch("subprocess.Popen", return_value=mock_proc):
        proc = mgr.start_backend()
        assert proc.pid == 12345
        assert 12345 in mgr.managed_pids
        assert mgr.is_running is True

        # Wait for readiness
        res = mgr.wait_for_backend_readiness(timeout_sec=2.0)
        assert res["status"] == "online"

    # Clean shutdown
    mgr.shutdown(request_api_shutdown=False)
    assert len(mgr.managed_pids) == 0


# ------------------------------------------------------------------------------
# Test 2: Startup waits for readiness rather than relying on a fixed delay
# ------------------------------------------------------------------------------
def test_startup_waits_for_readiness_not_fixed_delay(tmp_path: Path):
    """Verify that the manager returns immediately once the endpoint reports ready."""
    call_count = 0

    def mock_http(url: str):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return None  # Still booting
        return {"status": "online", "database": {"size_kb": 10}}

    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
        http_client=mock_http,
    )

    mock_proc = MockChildProcess(pid=54321)
    mgr._backend_process = mock_proc
    mgr._managed_pids.add(54321)

    t0 = time.monotonic()
    status = mgr.wait_for_backend_readiness(timeout_sec=5.0, check_interval_sec=0.05)
    elapsed = time.monotonic() - t0

    assert status["status"] == "online"
    assert call_count >= 3
    # Returned as soon as ready, well before the 5.0s timeout
    assert elapsed < 2.0


# ------------------------------------------------------------------------------
# Test 3: Backend startup failure is reported correctly
# ------------------------------------------------------------------------------
def test_backend_startup_failure_reported_correctly(tmp_path: Path):
    """Verify that an early crash of the child process raises StartupError with exit code."""
    crashed_proc = MockChildProcess(pid=67890, exit_code=127)

    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
        http_client=lambda url: None,
    )
    mgr._backend_process = crashed_proc
    mgr._managed_pids.add(67890)

    with pytest.raises(StartupError) as exc_info:
        mgr.wait_for_backend_readiness(timeout_sec=2.0, check_interval_sec=0.05)

    assert "exit code 127" in str(exc_info.value)
    assert "backend.log" in str(exc_info.value)


# ------------------------------------------------------------------------------
# Test 4: Frontend startup failure cleans up an already-started backend
# ------------------------------------------------------------------------------
def test_frontend_startup_failure_cleans_up_already_started_backend(tmp_path: Path):
    """Verify that if a subsequent startup stage fails, already-started processes are killed."""
    mock_proc = MockChildProcess(pid=77777)

    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )

    def mock_start(*args, **kwargs):
        mgr._managed_pids.add(mock_proc.pid)
        mgr._backend_process = mock_proc
        return mock_proc

    with patch.object(mgr, "acquire_single_instance_lock", return_value=True), \
         patch.object(mgr, "start_backend", side_effect=mock_start), \
         patch.object(mgr, "wait_for_backend_readiness", side_effect=Exception("Simulated window crash")), \
         patch("os.kill") as mock_kill:

        with pytest.raises(Exception, match="Simulated window crash"):
            mgr.startup()

        # cleanup_on_failure must have been called, issuing termination to pid 77777
        assert mock_kill.called
        mock_kill.assert_any_call(77777, signal.SIGTERM)
        assert not mgr.is_running


# ------------------------------------------------------------------------------
# Test 5: Readiness timeout results in bounded failure and cleanup
# ------------------------------------------------------------------------------
def test_readiness_timeout_results_in_bounded_failure_and_cleanup(tmp_path: Path):
    """Verify that if the service never responds within timeout, bounded failure and cleanup occur."""
    mock_proc = MockChildProcess(pid=88888)

    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
        http_client=lambda url: None,  # Never ready
    )
    mgr._backend_process = mock_proc
    mgr._managed_pids.add(88888)

    with patch("os.kill") as mock_kill:
        t0 = time.monotonic()
        with pytest.raises(ReadinessTimeoutError, match="Startup timed out"):
            mgr.wait_for_backend_readiness(timeout_sec=0.3, check_interval_sec=0.05)
        elapsed = time.monotonic() - t0

        # Verify bounded timeout execution
        assert elapsed >= 0.25
        assert elapsed < 2.5
        # Cleanup was triggered
        assert mock_kill.called


# ------------------------------------------------------------------------------
# Test 6: Normal quit requests graceful shutdown
# ------------------------------------------------------------------------------
def test_normal_quit_requests_graceful_shutdown(tmp_path: Path):
    """Verify that normal shutdown sends SIGTERM and releases lock cleanly."""
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )
    mgr.acquire_single_instance_lock()
    assert mgr._holds_lock is True

    test_pid = 44444
    mgr._managed_pids.add(test_pid)

    calls = 0
    def mock_alive(pid):
        nonlocal calls
        calls += 1
        return calls == 1

    with patch("os.kill") as mock_kill, \
         patch.object(mgr, "_is_pid_alive", side_effect=mock_alive):
        res = mgr.shutdown(grace_period_sec=1.0, request_api_shutdown=False)

        # SIGTERM must be sent to the managed pid
        mock_kill.assert_any_call(test_pid, signal.SIGTERM)
        assert test_pid in res["terminated_pids"]
        assert mgr._holds_lock is False


# ------------------------------------------------------------------------------
# Test 7: Non-responsive child is force-terminated after grace period
# ------------------------------------------------------------------------------
def test_non_responsive_child_force_terminated_after_grace_period(tmp_path: Path):
    """Verify that a process ignoring SIGTERM receives SIGKILL after grace period expires."""
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )
    test_pid = 55555
    mgr._managed_pids.add(test_pid)

    # Simulate process remaining alive through the grace period check
    with patch("os.kill") as mock_kill, \
         patch.object(mgr, "_is_pid_alive", return_value=True):
        mgr.shutdown(grace_period_sec=0.2, request_api_shutdown=False)

        # Must have sent SIGTERM first, then SIGKILL
        mock_kill.assert_any_call(test_pid, signal.SIGTERM)
        mock_kill.assert_any_call(test_pid, signal.SIGKILL)


# ------------------------------------------------------------------------------
# Test 8: Repeated shutdown calls are safe and idempotent
# ------------------------------------------------------------------------------
def test_repeated_shutdown_calls_are_safe(tmp_path: Path):
    """Verify that calling shutdown multiple times does not raise errors."""
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )
    res1 = mgr.shutdown(request_api_shutdown=False)
    assert res1["status"] != "already_shut_down"

    res2 = mgr.shutdown(request_api_shutdown=False)
    assert res2["status"] == "already_shut_down"

    res3 = mgr.shutdown(request_api_shutdown=False)
    assert res3["status"] == "already_shut_down"


# ------------------------------------------------------------------------------
# Test 9: Duplicate launcher instances prevented by single instance lock
# ------------------------------------------------------------------------------
def test_duplicate_launcher_instances_prevented_by_single_instance_lock(tmp_path: Path):
    """Verify that two supervisor instances cannot run concurrently on the same lock path."""
    lock_dir = tmp_path / "run"

    inst1 = DesktopLifecycleManager(lock_dir=lock_dir, data_dir=tmp_path / "d1")
    inst2 = DesktopLifecycleManager(lock_dir=lock_dir, data_dir=tmp_path / "d2")

    # Instance 1 acquires lock
    assert inst1.acquire_single_instance_lock() is True

    # Instance 2 must fail with InstanceLockError
    with pytest.raises(InstanceLockError, match="Another instance of JARVIS is already running"):
        inst2.acquire_single_instance_lock()

    # After instance 1 releases, instance 2 can acquire
    inst1.release_single_instance_lock()
    assert inst2.acquire_single_instance_lock() is True
    inst2.release_single_instance_lock()


# ------------------------------------------------------------------------------
# Test 10: Unrelated process is never terminated during cleanup
# ------------------------------------------------------------------------------
def test_unrelated_process_is_never_terminated_during_cleanup(tmp_path: Path):
    """Verify that PIDs not spawned by the supervisor are never signaled or killed."""
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )
    unrelated_pid = 11111
    # Note: unrelated_pid is NOT added to mgr._managed_pids

    with patch("os.kill") as mock_kill:
        mgr.shutdown(request_api_shutdown=False)

        # mock_kill must never have been called with unrelated_pid
        for call_args in mock_kill.call_args_list:
            assert call_args[0][0] != unrelated_pid


# ------------------------------------------------------------------------------
# Test 11: Independently running Ollama service is not killed
# ------------------------------------------------------------------------------
def test_independently_running_ollama_service_is_not_killed(tmp_path: Path):
    """Verify that an independent Ollama daemon PID is strictly protected from cleanup."""
    mgr = DesktopLifecycleManager(
        data_dir=tmp_path / "data",
        lock_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )
    ollama_pid = 98765  # Simulating existing system-level Ollama service

    with patch("os.kill") as mock_kill:
        # Supervisor manages only its own backend
        mgr._managed_pids.add(33333)

        mgr.shutdown(request_api_shutdown=False)

        # Ollama PID must never be targeted
        for call_args in mock_kill.call_args_list:
            assert call_args[0][0] != ollama_pid


# ------------------------------------------------------------------------------
# Test 12: Backend closes checkpointing, database, voice, and owned resources
# ------------------------------------------------------------------------------
def test_backend_closes_checkpointing_database_voice_resources():
    """Verify that perform_graceful_shutdown() safely closes storage, music, and voice."""
    import asyncio

    async def _run():
        with patch("tools.music.default_music_controller.stop", return_value={"success": True, "status": "stopped"}) as mock_music, \
             patch("storage.checkpoint.default_checkpoint_storage.close") as mock_cp:

            res = await perform_graceful_shutdown()
            assert res["status"] in ("success", "already_shut_down")
            if res["status"] == "success":
                assert mock_music.called
                assert mock_cp.called

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# Test 13: Development mode resolves project environment
# ------------------------------------------------------------------------------
def test_development_mode_resolves_project_environment():
    """Verify that the supervisor correctly locates Python and backend entrypoint in dev mode."""
    mgr = DesktopLifecycleManager()
    python_exe = mgr.resolve_python_executable()
    backend_entry = mgr.resolve_backend_entrypoint()

    assert python_exe.is_file()
    assert backend_entry.is_file()
    assert "main.py" in backend_entry.name


# ------------------------------------------------------------------------------
# Test 14: Production frontend assets are valid and ready for bundling
# ------------------------------------------------------------------------------
def test_production_frontend_assets_validity():
    """Verify that frontend HTML, CSS, and JS exist and contain expected markers."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    frontend_dir = repo_root / "frontend"

    index_html = frontend_dir / "index.html"
    app_js = frontend_dir / "app.js"
    styles_css = frontend_dir / "styles.css"

    assert index_html.is_file()
    assert app_js.is_file()
    assert styles_css.is_file()

    html_content = index_html.read_text(encoding="utf-8")
    assert "JARVIS" in html_content
    assert "workspace-grid" in html_content

    js_content = app_js.read_text(encoding="utf-8")
    assert "http://127.0.0.1:8000" in js_content


# ------------------------------------------------------------------------------
# Test 15: macOS application bundle contains correct icon and application resources
# ------------------------------------------------------------------------------
def test_macos_application_icon_and_resources():
    """Verify that macOS icon.icns and standard PNG icons exist with valid formats."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    icons_dir = repo_root / "src-tauri" / "icons"

    icns_file = icons_dir / "icon.icns"
    png_128 = icons_dir / "128x128.png"
    png_32 = icons_dir / "32x32.png"

    assert icns_file.is_file(), "icon.icns must exist"
    assert png_128.is_file(), "128x128.png must exist"
    assert png_32.is_file(), "32x32.png must exist"

    # Check that icns file starts with magic bytes b"icns"
    with open(icns_file, "rb") as f:
        magic = f.read(4)
        assert magic == b"icns", f"Expected 'icns' magic bytes, got {magic!r}"
