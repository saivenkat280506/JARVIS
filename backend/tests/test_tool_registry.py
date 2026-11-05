"""Unit tests for JARVIS Tool Registry and safe arithmetic evaluator."""
import pytest

from agent.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
    get_local_time,
    safe_calculate,
)


def test_safe_calculate_valid_expressions():
    assert safe_calculate("(125 * 8) + 40") == 1040
    assert safe_calculate("100 / 4") == 25.0
    assert safe_calculate("2 ** 10") == 1024
    assert safe_calculate("-(5 * 2) + 15") == 5
    assert safe_calculate("3.5 + 1.5") == 5.0
    assert safe_calculate("2 * (3 + 4 * (5 - 2))") == 30


def test_safe_calculate_division_by_zero():
    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        safe_calculate("100 / 0")


def test_safe_calculate_unsafe_syntax_rejection():
    # Code execution attempts
    with pytest.raises(ValueError):
        safe_calculate("import os")

    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        safe_calculate("eval('2+2')")

    with pytest.raises(ValueError):
        safe_calculate("exec('x = 1')")

    # Variables and function calls
    with pytest.raises(ValueError):
        safe_calculate("x + 1")

    with pytest.raises(ValueError):
        safe_calculate("print(42)")

    # Data structures and strings
    with pytest.raises(ValueError):
        safe_calculate("[1, 2, 3]")

    with pytest.raises(ValueError):
        safe_calculate("'hello' + 'world'")

    with pytest.raises(ValueError):
        safe_calculate("True + 1")


def test_safe_calculate_resource_limits():
    # Exponent too large
    with pytest.raises(ValueError, match="Exponent .* exceeds safe maximum"):
        safe_calculate("2 ** 1000")

    # Expression too long
    long_expr = "1 + " * 100 + "1"
    with pytest.raises(ValueError, match="maximum length"):
        safe_calculate(long_expr)

    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        safe_calculate("   ")


def test_get_local_time():
    result = get_local_time()
    assert isinstance(result, dict)
    assert "iso_timestamp" in result
    assert "readable_date" in result
    assert "readable_time" in result
    assert "timezone" in result
    assert len(result["timezone"]) > 0


def test_tool_registry_registration_and_schemas():
    registry = create_default_registry()
    tools = registry.list_tools()
    expected_tools = {
        "calculate",
        "get_local_time",
        "play_music",
        "pause_music",
        "resume_music",
        "stop_music",
        "open_website",
        "create_reminder",
        "list_reminders",
        "cancel_reminder",
        "browser_interact",
        "browser_observe",
    }
    assert expected_tools.issubset(set(tools))

    # Re-registering duplicate tool must fail
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            ToolDefinition(
                name="calculate",
                description="dup",
                parameters_schema={},
                handler=lambda: None,
            )
        )

    # Empty name must fail
    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register(
            ToolDefinition(
                name="",
                description="empty",
                parameters_schema={},
                handler=lambda: None,
            )
        )

    schemas = registry.get_schemas()
    assert len(schemas) == len(expected_tools)

    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


def test_tool_registry_execute_success(monkeypatch: pytest.MonkeyPatch):
    registry = create_default_registry()

    # Calculate execution
    res1 = registry.execute("calculate", {"expression": "(125 * 8) + 40"})
    assert res1["success"] is True
    assert res1["result"] == 1040

    # Local time execution
    res2 = registry.execute("get_local_time", {})
    assert res2["success"] is True
    assert "readable_date" in res2["result"]

    # Open website execution (mock launcher)
    import subprocess
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "tools.browser.default_browser_controller._launcher",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, returncode=0),
    )
    res3 = registry.execute("open_website", {"url": "apple.com"})
    assert res3["success"] is True
    assert res3["result"]["url"] == "https://apple.com"


def test_tool_registry_execute_unknown_tool():
    registry = create_default_registry()
    res = registry.execute("system_shutdown", {})
    assert res["success"] is False
    assert "is not registered" in res["error"]
    assert "available_tools" in res



def test_tool_registry_execute_invalid_arguments():
    registry = create_default_registry()

    # Missing required argument 'expression'
    res_missing = registry.execute("calculate", {})
    assert res_missing["success"] is False
    assert "Invalid arguments" in res_missing["error"]

    # Extra disallowed argument
    res_extra = registry.execute(
        "calculate", {"expression": "10 + 2", "malicious_arg": "hack"}
    )
    assert res_extra["success"] is False
    assert "Invalid arguments" in res_extra["error"]

    # Extra disallowed argument on no-args tool
    res_time_extra = registry.execute("get_local_time", {"unexpected": True})
    assert res_time_extra["success"] is False
    assert "Invalid arguments" in res_time_extra["error"]

    # Invalid argument types
    res_type = registry.execute("calculate", {"expression": 12345})
    assert res_type["success"] is False
    assert "Invalid arguments" in res_type["error"]


def test_tool_registry_execute_math_error_handling():
    registry = create_default_registry()
    res = registry.execute("calculate", {"expression": "10 / 0"})
    assert res["success"] is False
    assert "Division by zero" in res["error"]
