"""Chrome DevTools Protocol (CDP) client for secure local browser interaction."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx
import websockets

from config.settings import settings

logger = logging.getLogger(__name__)

CHROME_LAUNCH_HINT = (
    "Chrome remote debugging session is unavailable on {host}:{port}. "
    "To enable browser automation, start Chrome with: "
    "'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --remote-debugging-port={port}'"
)


class CDPConnectionError(Exception):
    """Raised when unable to connect to the Chrome DevTools debugging port."""


class CDPCommandError(Exception):
    """Raised when a CDP command returns an error from the browser."""


class CDPClient:
    """Async client communicating with local Chrome via HTTP and WebSocket DevTools Protocol."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> None:
        self.host = host or settings.cdp_host
        self.port = port or settings.cdp_port
        self.timeout_sec = timeout_sec or settings.cdp_timeout_sec
        self.http_base_url = f"http://{self.host}:{self.port}"

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_target_id: Optional[str] = None
        self._msg_id = 0
        self._pending_commands: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def is_available(self) -> Tuple[bool, str]:
        """Check if Chrome remote debugging port is open and responding.

        Returns:
            Tuple[bool, str]: (True, "") if available, (False, instruction_message) if not.
        """
        url = f"{self.http_base_url}/json/version"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True, ""
                return False, CHROME_LAUNCH_HINT.format(host=self.host, port=self.port)
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False, CHROME_LAUNCH_HINT.format(host=self.host, port=self.port)

    async def list_targets(self) -> List[Dict[str, Any]]:
        """List active targets (tabs) in Chrome."""
        url = f"{self.http_base_url}/json/list"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            raise CDPConnectionError(
                f"Failed to query Chrome targets: {exc}. "
                + CHROME_LAUNCH_HINT.format(host=self.host, port=self.port)
            ) from exc

    async def get_or_create_page_target(self) -> Dict[str, Any]:
        """Get an existing page target or open a new one."""
        targets = await self.list_targets()
        page_targets = [t for t in targets if t.get("type") == "page"]

        if page_targets:
            return page_targets[0]

        # No page targets open, create a blank tab
        url = f"{self.http_base_url}/json/new"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.put(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            raise CDPConnectionError(f"Failed to create new page target: {exc}") from exc

    @property
    def is_connected(self) -> bool:
        """Check whether the WebSocket connection is open and active."""
        if self._ws is None:
            return False
        state = getattr(self._ws, "state", None)
        if state is not None:
            return getattr(state, "name", "").upper() == "OPEN"
        return not getattr(self._ws, "closed", True)

    async def connect_to_target(self, target_id: Optional[str] = None) -> None:
        """Establish WebSocket connection to the designated page target."""
        async with self._lock:
            if self.is_connected:
                if target_id is None or target_id == self._current_target_id:
                    return
                await self._disconnect_ws()

            targets = await self.list_targets()
            page_targets = [t for t in targets if t.get("type") == "page"]

            target = None
            if target_id:
                for t in page_targets:
                    if t.get("id") == target_id:
                        target = t
                        break
                if not target:
                    raise CDPConnectionError(f"Target page '{target_id}' not found.")
            else:
                if page_targets:
                    target = page_targets[0]
                else:
                    target = await self.get_or_create_page_target()

            ws_url = target.get("webSocketDebuggerUrl")
            if not ws_url:
                raise CDPConnectionError("Target does not provide a webSocketDebuggerUrl.")

            try:
                self._ws = await websockets.connect(
                    ws_url,
                    ping_interval=None,
                    close_timeout=2.0,
                )
                self._current_target_id = target.get("id")
                self._reader_task = asyncio.create_task(self._reader_loop())
            except Exception as exc:
                raise CDPConnectionError(f"Failed to connect to page WebSocket '{ws_url}': {exc}") from exc

    async def _reader_loop(self) -> None:
        """Background reader receiving incoming WebSocket messages."""
        try:
            while self.is_connected:
                msg_str = await self._ws.recv()
                msg = json.loads(msg_str)
                msg_id = msg.get("id")

                if msg_id is not None and msg_id in self._pending_commands:
                    fut = self._pending_commands.pop(msg_id)
                    if not fut.done():
                        fut.set_result(msg)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.debug("CDP reader loop encountered exception: %s", exc)

    async def send_command(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC command over CDP WebSocket and return the result dictionary."""
        if not self.is_connected:
            await self.connect_to_target(self._current_target_id)

        timeout_duration = timeout or self.timeout_sec
        async with self._lock:
            self._msg_id += 1
            cmd_id = self._msg_id
            payload = {
                "id": cmd_id,
                "method": method,
                "params": params or {},
            }
            fut = asyncio.get_running_loop().create_future()
            self._pending_commands[cmd_id] = fut

            try:
                await self._ws.send(json.dumps(payload))
            except Exception as exc:
                self._pending_commands.pop(cmd_id, None)
                raise CDPConnectionError(f"Failed to send command '{method}': {exc}") from exc

        try:
            response = await asyncio.wait_for(fut, timeout=timeout_duration)
        except asyncio.TimeoutError:
            self._pending_commands.pop(cmd_id, None)
            raise CDPCommandError(f"CDP command '{method}' timed out after {timeout_duration}s.")

        if "error" in response:
            err = response["error"]
            raise CDPCommandError(f"CDP error {err.get('code')}: {err.get('message')}")

        return response.get("result", {})

    async def _disconnect_ws(self) -> None:
        """Close WebSocket connection and reader task."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for fut in self._pending_commands.values():
            if not fut.done():
                fut.cancel()
        self._pending_commands.clear()
        self._current_target_id = None

    async def close(self) -> None:
        """Cleanly close all open CDP connections."""
        async with self._lock:
            await self._disconnect_ws()


# Shared instance
default_cdp_client = CDPClient()
