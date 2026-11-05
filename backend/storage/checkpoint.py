"""Durable SQLite checkpoint storage for LangGraph agent workflows.

Provides reliable, process-persistent checkpoint storage in data/checkpoints.db:
- Uses LangGraph's AsyncSqliteSaver backed by aiosqlite.
- Safe lifecycle management (setup, retrieval, and closing).
- Avoids duplicate database connections or competing savers.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config.settings import settings

logger = logging.getLogger(__name__)


class CheckpointStorage:
    """Manages a shared, persistent AsyncSqliteSaver checkpointer for LangGraph."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = getattr(
                settings,
                "checkpoint_database_path",
                settings.data_dir / "checkpoints.db",
            )
        self._cm: Optional[Any] = None
        self._saver: Optional[AsyncSqliteSaver] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock: Optional[asyncio.Lock] = None
        self._is_closed = False

    @property
    def saver(self) -> Optional[AsyncSqliteSaver]:
        """Return the current initialized AsyncSqliteSaver, if any."""
        return self._saver

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop != loop:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_or_create_saver(self) -> AsyncSqliteSaver:
        """Initialize and return the shared AsyncSqliteSaver checkpointer for current event loop."""
        loop = asyncio.get_running_loop()
        if (
            self._saver is not None
            and self._loop == loop
            and not self._is_closed
        ):
            # Confirm the underlying connection is still open
            if getattr(self._saver, "conn", None) is not None:
                try:
                    _ = self._saver.conn._conn
                    return self._saver
                except Exception:
                    pass

        lock = self._get_lock()
        async with lock:
            if (
                self._saver is not None
                and self._loop == loop
                and not self._is_closed
            ):
                if getattr(self._saver, "conn", None) is not None:
                    try:
                        _ = self._saver.conn._conn
                        return self._saver
                    except Exception:
                        pass

            # Close previous context manager if same loop or stale
            if self._cm is not None:
                try:
                    if self._loop == loop:
                        await self._cm.__aexit__(None, None, None)
                except Exception:
                    pass
                self._cm = None
                self._saver = None

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cm = AsyncSqliteSaver.from_conn_string(str(self.db_path))
            self._saver = await self._cm.__aenter__()
            await self._saver.setup()
            self._loop = loop
            self._is_closed = False
            logger.info("Initialized LangGraph SQLite checkpointer at %s (loop %s)", self.db_path, id(loop))
            return self._saver

    async def close(self) -> None:
        """Safely close the checkpoint connection."""
        lock = self._get_lock()
        async with lock:
            if self._cm is not None and not self._is_closed:
                try:
                    await self._cm.__aexit__(None, None, None)
                except Exception as exc:
                    logger.warning("Error closing checkpointer connection: %s", exc)
                finally:
                    self._saver = None
                    self._cm = None
                    self._loop = None
                    self._is_closed = True


# Global shared instance using the project-wide checkpoint database
default_checkpoint_storage = CheckpointStorage()
