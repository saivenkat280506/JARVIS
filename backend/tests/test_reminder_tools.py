"""Unit tests for reminder tool handlers and agent runtime integration."""
from pathlib import Path
import pytest

from agent.runtime import AgentRuntime
from agent.tool_registry import create_default_registry
from storage.reminders import ReminderStorage
from tests.test_agent_runtime import MockOllamaClient


@pytest.fixture
def temp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_db = tmp_path / "tools_test.db"
    storage = ReminderStorage(db_path=test_db)
    monkeypatch.setattr("tools.reminders.default_reminder_storage", storage)
    return storage


def test_create_reminder_tool_success(temp_storage: ReminderStorage):
    registry = create_default_registry()

    res = registry.execute(
        "create_reminder",
        {"text": "Call the mechanic", "due_time": "in 15 minutes"},
    )
    assert res["success"] is True
    assert res["result"]["success"] is True
    assert "rem_" in res["result"]["reminder"]["id"]

    # Verify directly in SQLite
    listed = temp_storage.list_reminders()
    assert len(listed) == 1
    assert listed[0].text == "Call the mechanic"


def test_create_reminder_tool_ambiguous_rejection(temp_storage: ReminderStorage):
    registry = create_default_registry()

    res = registry.execute(
        "create_reminder",
        {"text": "Submit report", "due_time": "later"},
    )
    assert res["success"] is True
    assert res["result"]["success"] is False
    assert "Ambiguous due time" in res["result"]["error"]


def test_create_reminder_tool_extra_args_rejected(temp_storage: ReminderStorage):
    registry = create_default_registry()

    res = registry.execute(
        "create_reminder",
        {
            "text": "Submit report",
            "due_time": "in 5 minutes",
            "malicious_extra": "drop table reminders;",
        },
    )
    assert res["success"] is False
    assert "Invalid arguments" in res["error"]


def test_list_and_cancel_reminder_tools(temp_storage: ReminderStorage):
    registry = create_default_registry()

    # Create reminder
    res_create = registry.execute(
        "create_reminder",
        {"text": "Buy groceries", "due_time": "in 30 minutes"},
    )
    rem_id = res_create["result"]["reminder"]["id"]

    # List
    res_list = registry.execute("list_reminders", {"status": "pending"})
    assert res_list["success"] is True
    assert res_list["result"]["count"] == 1

    # Cancel
    res_cancel = registry.execute("cancel_reminder", {"reminder_id": rem_id})
    assert res_cancel["success"] is True
    assert res_cancel["result"]["success"] is True

    # Cancel non-existent
    res_cancel_bad = registry.execute(
        "cancel_reminder", {"reminder_id": "rem_nonexistent"}
    )
    assert res_cancel_bad["success"] is True
    assert res_cancel_bad["result"]["success"] is False


@pytest.mark.anyio
async def test_runtime_reminder_creation_turn(
    temp_storage: ReminderStorage, monkeypatch: pytest.MonkeyPatch
):
    mock_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_rem_1",
                    "function": {
                        "name": "create_reminder",
                        "arguments": {
                            "text": "Review budget spreadsheet",
                            "due_time": "in 2 hours",
                        },
                    },
                }],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "I have created your reminder to review the budget spreadsheet in 2 hours, sir.",
            }
        },
    ]

    client = MockOllamaClient(mock_responses)
    registry = create_default_registry()
    runtime = AgentRuntime(client=client, registry=registry)

    result = await runtime.run("Remind me to review budget spreadsheet in 2 hours")

    assert result.completed is True
    assert result.status == "success"
    assert len(result.tool_records) == 1
    assert result.tool_records[0].tool_name == "create_reminder"
    assert result.tool_records[0].result["success"] is True
    assert result.tool_records[0].result["result"]["success"] is True

    # Verify reminder exists in database
    reminders = temp_storage.list_reminders()
    assert len(reminders) == 1
    assert reminders[0].text == "Review budget spreadsheet"
