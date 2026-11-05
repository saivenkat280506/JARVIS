"""Lifecycle Manager and Process Supervisor for JARVIS Desktop Application.

Provides robust process lifecycle management:
1. Single-instance acquisition via OS advisory lockfile.
2. Reliable Python runtime and backend discovery.
3. Process-group tracked child execution.
4. Bounded HTTP readiness polling with early crash detection.
5. Strict process ownership (only owns and stops explicitly launched PIDs;
   NEVER kills processes by name or port, protecting independent Ollama instances).
6. Graceful shutdown sequence: SIGTERM / HTTP /api/shutdown -> bounded grace period -> SIGKILL.
7. Zombie reaping and file handle cleanup.
8. Idempotent shutdown and atomic failure rollback.
"""
from __future__ import annotations

import errno
import fcntl
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class InstanceLockError(Exception):
    """Raised when another managed instance of JARVIS desktop is already running."""


class StartupError(Exception):
    """Raised when an application startup stage fails."""


class PortConflictError(StartupError):
    """Raised when a required network port is already in use by an unmanaged process."""


class ReadinessTimeoutError(StartupError):
    """Raised when managed services fail to become ready within the bounded timeout."""


class DesktopLifecycleManager:
    """Process supervisor and lifecycle manager for JARVIS desktop."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        lock_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        backend_host: str = "127.0.0.1",
        backend_port: int = 8000,
        grace_period_sec: float = 3.0,
        readiness_timeout_sec: float = 15.0,
        http_client: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.project_root = (
            project_root or Path(__file__).resolve().parent.parent.parent
        )
        self.backend_dir = self.project_root / "backend"

        # Determine user data directory (defaults to ~/Library/Application Support/JARVIS on macOS)
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        elif os.environ.get("JARVIS_DATA_DIR"):
            self.data_dir = Path(os.environ["JARVIS_DATA_DIR"])
        elif sys.platform == "darwin":
            self.data_dir = (
                Path.home() / "Library" / "Application Support" / "JARVIS" / "data"
            )
        else:
            self.data_dir = Path.home() / ".jarvis" / "data"

        # Determine lock and log directories
        self.lock_dir = (
            Path(lock_dir) if lock_dir is not None else (self.data_dir.parent / "run")
        )
        self.log_dir = (
            Path(log_dir) if log_dir is not None else (self.data_dir.parent / "logs")
        )

        self.backend_host = backend_host
        self.backend_port = backend_port
        self.grace_period_sec = grace_period_sec
        self.readiness_timeout_sec = readiness_timeout_sec

        # Single instance lock state
        self._lock_file_path = self.lock_dir / "jarvis.lock"
        self._lock_file_fd: Optional[int] = None
        self._holds_lock = False

        # Managed process tracking (strictly records spawned PIDs)
        self._managed_pids: Set[int] = set()
        self._managed_processes: List[subprocess.Popen] = []
        self._backend_process: Optional[subprocess.Popen] = None
        self._backend_log_file = None

        # State flags
        self._is_shut_down = False
        self._http_client = http_client

    @property
    def managed_pids(self) -> Set[int]:
        """Return a copy of all child PIDs currently owned by this supervisor."""
        return set(self._managed_pids)

    @property
    def is_running(self) -> bool:
        """Return True if backend process is currently active."""
        if self._backend_process is not None:
            return self._backend_process.poll() is None
        return False

    # --------------------------------------------------------------------------
    # 1. Single Instance Ownership
    # --------------------------------------------------------------------------
    def acquire_single_instance_lock(self) -> bool:
        """Acquire non-blocking advisory file lock for the application instance.

        Returns True if successfully acquired; raises InstanceLockError if another
        instance is already active.
        """
        if self._holds_lock:
            return True

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                str(self._lock_file_path),
                os.O_CREAT | os.O_RDWR,
                0o644,
            )
        except OSError as exc:
            raise InstanceLockError(
                f"Could not open lockfile at {self._lock_file_path}: {exc}"
            ) from exc

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            # Read existing owner PID if present for diagnostics
            existing_pid = "unknown"
            try:
                if self._lock_file_path.is_file():
                    existing_pid = self._lock_file_path.read_text().strip()
            except Exception:
                pass
            raise InstanceLockError(
                f"Another instance of JARVIS is already running (PID {existing_pid}). "
                "Only one managed application instance is permitted."
            ) from exc

        # Lock successfully acquired; write current PID
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to write PID into lockfile: %s", exc)

        self._lock_file_fd = fd
        self._holds_lock = True
        logger.info(
            "Acquired single-instance lock at %s (PID %d)",
            self._lock_file_path,
            os.getpid(),
        )
        return True

    def release_single_instance_lock(self) -> None:
        """Release the advisory file lock cleanly."""
        if self._lock_file_fd is not None:
            try:
                fcntl.flock(self._lock_file_fd, fcntl.LOCK_UN)
                os.close(self._lock_file_fd)
            except Exception as exc:
                logger.warning("Error releasing lockfile fd: %s", exc)
            finally:
                self._lock_file_fd = None
                self._holds_lock = False

        try:
            if self._lock_file_path.is_file():
                self._lock_file_path.unlink()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 2. Port Conflict and Dependency Checks
    # --------------------------------------------------------------------------
    def is_port_in_use(self, host: str, port: int) -> bool:
        """Check whether a TCP port is currently occupied by an active process."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex((host, port))
            return result == 0

    def verify_port_available(self, host: str, port: int) -> None:
        """Verify that the target service port is free before attempting startup."""
        if self.is_port_in_use(host, port):
            raise PortConflictError(
                f"Port {port} on {host} is already in use by an existing process. "
                "JARVIS Bridge cannot bind to port 8000. Please resolve the port conflict."
            )

    # --------------------------------------------------------------------------
    # 3. Python Runtime and Entrypoint Discovery
    # --------------------------------------------------------------------------
    def resolve_python_executable(self) -> Path:
        """Discover and validate the appropriate Python runtime."""
        # 1. Explicit override via environment
        env_python = os.environ.get("JARVIS_PYTHON")
        if env_python and Path(env_python).is_file():
            return Path(env_python).resolve()

        # 2. Virtual environment in project root
        venv_python = self.project_root / ".venv" / "bin" / "python"
        if venv_python.is_file():
            return venv_python.resolve()

        # 3. Virtual environment in user home
        user_venv = Path.home() / ".jarvis" / "venv" / "bin" / "python"
        if user_venv.is_file():
            return user_venv.resolve()

        # 4. Current running Python interpreter
        if Path(sys.executable).is_file():
            return Path(sys.executable).resolve()

        raise StartupError(
            "Could not locate a valid Python interpreter. "
            "Please ensure JARVIS virtual environment is created (.venv) or set JARVIS_PYTHON."
        )

    def resolve_backend_entrypoint(self) -> Path:
        """Locate the Python backend entrypoint script."""
        # Check backend/main.py
        candidate = self.backend_dir / "main.py"
        if candidate.is_file():
            return candidate.resolve()

        # Fallback to root main.py
        candidate = self.project_root / "main.py"
        if candidate.is_file():
            return candidate.resolve()

        raise StartupError(
            f"Backend entrypoint not found in {self.backend_dir} or {self.project_root}."
        )

    # --------------------------------------------------------------------------
    # 4. Managed Backend Process Spawning
    # --------------------------------------------------------------------------
    def start_backend(
        self,
        extra_env: Optional[Dict[str, str]] = None,
        custom_launcher: Optional[Callable[..., subprocess.Popen]] = None,
    ) -> subprocess.Popen:
        """Launch the FastAPI backend bridge as an explicitly tracked child process."""
        self.verify_port_available(self.backend_host, self.backend_port)

        python_bin = self.resolve_python_executable()
        entry_script = self.resolve_backend_entrypoint()

        # Prepare log directory and file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = self.log_dir / "backend.log"
        self._backend_log_file = open(str(log_file_path), "a", encoding="utf-8")

        # Environment variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["JARVIS_DATA_DIR"] = str(self.data_dir)
        env["PORT"] = str(self.backend_port)
        env["HOST"] = str(self.backend_host)
        if extra_env:
            env.update(extra_env)

        cmd = [str(python_bin), str(entry_script), "--bridge"]
        logger.info("Spawning managed backend process: %s", " ".join(cmd))

        launcher = custom_launcher or subprocess.Popen
        try:
            # On POSIX systems, start in a new process group so child subprocesses
            # can be tracked and terminated reliably if needed.
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = launcher(
                cmd,
                cwd=str(self.backend_dir),
                env=env,
                stdout=self._backend_log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=preexec,
            )
        except Exception as exc:
            if self._backend_log_file:
                self._backend_log_file.close()
                self._backend_log_file = None
            raise StartupError(f"Failed to spawn backend process: {exc}") from exc

        # Track process ownership
        self._backend_process = proc
        self._managed_pids.add(proc.pid)
        self._managed_processes.append(proc)
        logger.info(
            "Backend process launched successfully with PID %d (Log: %s)",
            proc.pid,
            log_file_path,
        )
        return proc

    # --------------------------------------------------------------------------
    # 5. Service Readiness Polling
    # --------------------------------------------------------------------------
    def wait_for_backend_readiness(
        self,
        endpoint_url: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        check_interval_sec: float = 0.2,
    ) -> Dict[str, Any]:
        """Poll the backend status/diagnostics endpoint until ready or timeout.

        Immediately aborts if the child process terminates prematurely.
        """
        url = (
            endpoint_url
            or f"http://{self.backend_host}:{self.backend_port}/api/status"
        )
        timeout = (
            timeout_sec if timeout_sec is not None else self.readiness_timeout_sec
        )
        start_time = time.monotonic()

        logger.info("Waiting for backend readiness at %s (timeout: %.1fs)...", url, timeout)

        while (time.monotonic() - start_time) < timeout:
            # Check if backend died unexpectedly
            if self._backend_process is not None:
                exit_code = self._backend_process.poll()
                if exit_code is not None:
                    # Child process crashed
                    self.cleanup_on_failure()
                    raise StartupError(
                        f"Backend process terminated unexpectedly during startup with exit code {exit_code}. "
                        f"Check logs at {self.log_dir / 'backend.log'} for details."
                    )

            # Query endpoint
            try:
                data = self._check_http_status(url)
                if data:
                    elapsed = time.monotonic() - start_time
                    logger.info(
                        "Backend service ready in %.2fs (status: %s)",
                        elapsed,
                        data.get("status", "ok"),
                    )
                    return data
            except Exception:
                pass

            time.sleep(check_interval_sec)

        # Timeout reached
        self.cleanup_on_failure()
        raise ReadinessTimeoutError(
            f"Backend did not report ready within {timeout:.1f}s at {url}. "
            f"Startup timed out. Check logs at {self.log_dir / 'backend.log'}."
        )

    def _check_http_status(self, url: str) -> Optional[Dict[str, Any]]:
        """Query HTTP endpoint using custom client or urllib with short timeout."""
        if self._http_client is not None:
            return self._http_client(url)

        import json
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "JARVIS-Supervisor"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
        except Exception:
            return None
        return None

    # --------------------------------------------------------------------------
    # 6. Complete Startup Sequence
    # --------------------------------------------------------------------------
    def startup(self) -> Dict[str, Any]:
        """Execute complete application startup lifecycle.

        1. Acquire single-instance lock.
        2. Verify port availability.
        3. Start managed backend child process.
        4. Bounded wait for service readiness.
        """
        logger.info("Beginning JARVIS application startup...")
        try:
            self.acquire_single_instance_lock()
            proc = self.start_backend()
            status = self.wait_for_backend_readiness()
            return {
                "status": "ready",
                "backend_pid": proc.pid,
                "backend_status": status,
            }
        except Exception as exc:
            logger.error("Startup sequence failed: %s", exc)
            self.cleanup_on_failure()
            raise

    # --------------------------------------------------------------------------
    # 7. Safe Process Termination and Complete Shutdown
    # --------------------------------------------------------------------------
    def shutdown(
        self,
        grace_period_sec: Optional[float] = None,
        request_api_shutdown: bool = True,
    ) -> Dict[str, Any]:
        """Safely terminate all managed processes and clean up resources.

        Guarantees:
        - Only touches PIDs recorded in self._managed_pids.
        - NEVER terminates external processes (such as Ollama or user daemons).
        - Requests graceful HTTP shutdown first if applicable.
        - Sends SIGTERM with bounded grace period before SIGKILL.
        - Reaps zombie child processes.
        - Releases lockfile.
        - Idempotent and safe to invoke multiple times.
        """
        if self._is_shut_down:
            return {"status": "already_shut_down", "terminated_pids": []}

        grace = (
            grace_period_sec if grace_period_sec is not None else self.grace_period_sec
        )
        logger.info("Executing JARVIS supervisor shutdown (grace period: %.1fs)...", grace)

        results: Dict[str, Any] = {
            "status": "completed",
            "terminated_pids": [],
            "failed_pids": [],
            "api_shutdown_success": False,
        }

        # 1. Optionally request graceful shutdown via REST bridge
        if request_api_shutdown and self.is_running:
            shutdown_url = (
                f"http://{self.backend_host}:{self.backend_port}/api/shutdown"
            )
            try:
                import urllib.request

                req = urllib.request.Request(
                    shutdown_url,
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        results["api_shutdown_success"] = True
                        logger.info("Graceful backend shutdown requested via /api/shutdown.")
            except Exception as exc:
                logger.debug("Bridge /api/shutdown call was not acknowledged: %s", exc)

        # 2. Iterate strictly over managed PIDs
        target_pids = list(self._managed_pids)
        for pid in target_pids:
            if not self._is_pid_alive(pid):
                results["terminated_pids"].append(pid)
                continue

            logger.info("Terminating owned child PID %d with SIGTERM...", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                results["terminated_pids"].append(pid)
                continue
            except OSError as exc:
                logger.warning("Error sending SIGTERM to PID %d: %s", pid, exc)

        # 3. Wait up to grace period for graceful termination
        deadline = time.monotonic() + grace
        while (time.monotonic() < deadline) and any(
            self._is_pid_alive(p) for p in target_pids
        ):
            time.sleep(0.1)

        # 4. Force kill (SIGKILL) any remaining owned PIDs
        for pid in target_pids:
            if self._is_pid_alive(pid):
                logger.warning(
                    "Child PID %d did not terminate gracefully within %.1fs; sending SIGKILL...",
                    pid,
                    grace,
                )
                try:
                    os.kill(pid, signal.SIGKILL)
                    results["terminated_pids"].append(pid)
                except ProcessLookupError:
                    results["terminated_pids"].append(pid)
                except OSError as exc:
                    logger.error("Failed to SIGKILL child PID %d: %s", pid, exc)
                    results["failed_pids"].append(pid)
            else:
                if pid not in results["terminated_pids"]:
                    results["terminated_pids"].append(pid)

        # 5. Reap managed processes to prevent zombies
        for proc in self._managed_processes:
            try:
                proc.poll()
                # Non-blocking wait if already dead or quick wait
                proc.wait(timeout=0.5)
            except Exception:
                pass

        # 6. Close logging file handles
        if self._backend_log_file is not None:
            try:
                self._backend_log_file.flush()
                self._backend_log_file.close()
            except Exception:
                pass
            self._backend_log_file = None

        # 7. Release lockfile
        self.release_single_instance_lock()

        # Update state
        self._managed_pids.clear()
        self._managed_processes.clear()
        self._backend_process = None
        self._is_shut_down = True

        logger.info("JARVIS supervisor shutdown completed successfully.")
        return results

    def cleanup_on_failure(self) -> None:
        """Rollback all partially started resources if startup fails."""
        logger.info("Performing failure cleanup on managed resources...")
        try:
            self.shutdown(grace_period_sec=1.0, request_api_shutdown=False)
        except Exception as exc:
            logger.warning("Error during failure cleanup: %s", exc)

    def _is_pid_alive(self, pid: int) -> bool:
        """Check whether a PID is alive without sending a terminating signal."""
        if pid <= 0:
            return False
        try:
            # Signal 0 performs error checking but does not send a signal
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but owned by someone else; we shouldn't kill it anyway
            return True
        except OSError:
            return False


def main() -> int:
    """Run desktop lifecycle manager directly from command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    mgr = DesktopLifecycleManager()

    def handle_sig(sig, frame):
        print(f"\nReceived signal {sig}; shutting down JARVIS desktop services gracefully...")
        mgr.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        res = mgr.startup()
        print(f"JARVIS Desktop services started successfully (PID: {res.get('backend_pid')}).")
        print("Backend running at http://127.0.0.1:8000. Press Ctrl+C to stop.")
        while mgr.is_running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[Error] Desktop startup failed: {exc}", file=sys.stderr)
        mgr.cleanup_on_failure()
        return 1
    finally:
        mgr.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

