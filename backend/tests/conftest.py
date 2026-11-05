"""Pytest configuration and strict database isolation for JARVIS test suite.

Ensures that:
1. DATABASE_PATH environment variable is configured to an isolated temporary SQLite database
   BEFORE any application modules or singletons are imported.
2. Production database (data/jarvis.db) is never targeted by tests or import-time recovery.
3. An autouse guard asserts that neither settings.database_path nor any global singleton
   resolves to the production database path.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import pytest

# 1. Establish isolated session database BEFORE any application imports
_PROD_PATH = Path(__file__).resolve().parent.parent / "data" / "jarvis.db"

if "DATABASE_PATH" not in os.environ or Path(os.environ["DATABASE_PATH"]).resolve() == _PROD_PATH.resolve():
    _test_db_dir = tempfile.mkdtemp(prefix="jarvis_test_session_")
    _session_test_db = Path(_test_db_dir) / "test_jarvis.db"
    os.environ["DATABASE_PATH"] = str(_session_test_db)

# 2. Configure settings.database_path immediately
from config.settings import DATABASE_PATH as CANONICAL_PROD_DB_PATH, settings
settings.database_path = Path(os.environ["DATABASE_PATH"]).resolve()

# Ensure singletons are imported after settings.database_path is isolated
from security.approval import default_approval_manager
from storage.action_ledger import default_action_ledger
from storage.conversation import default_conversation_storage
from storage.reminders import default_reminder_storage
from storage.tasks import default_task_storage


@pytest.fixture(autouse=True)
def guard_against_production_database() -> None:
    """Fail immediately if any test or singleton attempts to touch production database."""
    prod_resolved = CANONICAL_PROD_DB_PATH.resolve()

    assert settings.database_path.resolve() != prod_resolved, (
        f"CRITICAL: settings.database_path resolved to production database '{prod_resolved}'!"
    )

    managers = {
        "default_approval_manager": default_approval_manager,
        "default_action_ledger": default_action_ledger,
        "default_conversation_storage": default_conversation_storage,
        "default_reminder_storage": default_reminder_storage,
        "default_task_storage": default_task_storage,
    }

    for name, mgr in managers.items():
        assert mgr.db_path.resolve() != prod_resolved, (
            f"CRITICAL: {name}.db_path resolved to production database '{prod_resolved}'!"
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up temporary test database directory after test session completes."""
    db_env = os.environ.get("DATABASE_PATH")
    if db_env:
        p = Path(db_env)
        if "jarvis_test_session_" in str(p.parent):
            shutil.rmtree(p.parent, ignore_errors=True)
