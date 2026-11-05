"""Unit tests for BrowserController and URL normalization."""
import subprocess
from typing import Any, List
import pytest

from tools.browser import BrowserController, normalize_and_validate_url


def test_url_normalization_valid():
    assert normalize_and_validate_url("google.com") == "https://google.com"
    assert (
        normalize_and_validate_url("apple.com/macbook-pro")
        == "https://apple.com/macbook-pro"
    )
    assert (
        normalize_and_validate_url("https://news.ycombinator.com/item?id=12345")
        == "https://news.ycombinator.com/item?id=12345"
    )
    assert (
        normalize_and_validate_url("http://localhost:3000/app")
        == "http://localhost:3000/app"
    )
    assert (
        normalize_and_validate_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
        == "https://en.wikipedia.org/wiki/Artificial_intelligence"
    )


def test_url_rejection_forbidden_schemes():
    forbidden_examples = [
        "file:///etc/passwd",
        "file:///Users/saivenkat/.bash_profile",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "vbscript:msgbox(1)",
        "about:blank",
        "ftp://ftp.example.com/file.zip",
        "ssh://git@github.com",
        "ws://websocket.example.com",
        "wss://secure.example.com",
    ]
    for url in forbidden_examples:
        with pytest.raises(ValueError, match="forbidden|permitted"):
            normalize_and_validate_url(url)



def test_url_rejection_shell_syntax():
    shell_attacks = [
        "https://google.com; rm -rf /",
        "https://google.com | cat /etc/passwd",
        "https://google.com && echo pwned",
        "https://google.com/test$(whoami)",
        "https://google.com/test`id`",
        "https://google.com\nhttp://attacker.com",
        "https://google.com\rhttp://attacker.com",
        "https://google.com<malicious>",
    ]
    for attack in shell_attacks:
        with pytest.raises(ValueError, match="disallowed syntax|shell control"):
            normalize_and_validate_url(attack)


def test_url_rejection_embedded_credentials():
    with pytest.raises(ValueError, match="Embedded credentials"):
        normalize_and_validate_url("https://admin:secret@example.com")

    with pytest.raises(ValueError, match="Embedded credentials"):
        normalize_and_validate_url("http://user@example.com/dashboard")


def test_url_rejection_malformed_hostnames():
    # Missing hostname
    with pytest.raises(ValueError, match="valid network hostname"):
        normalize_and_validate_url("https://")

    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_and_validate_url("   ")

    # Excessive length
    huge_url = "https://example.com/" + ("a" * 2050)
    with pytest.raises(ValueError, match="maximum length"):
        normalize_and_validate_url(huge_url)

    # Whitespace
    with pytest.raises(ValueError, match="disallowed syntax"):
        normalize_and_validate_url("https://example .com")


def test_browser_controller_dispatch_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)

    launched_commands: List[List[str]] = []

    def mock_launcher(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        launched_commands.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    controller = BrowserController(
        open_binary="/usr/bin/open",
        launcher=mock_launcher,
    )

    result = controller.open_website("google.com")
    assert result["success"] is True
    assert result["url"] == "https://google.com"
    assert result["dispatched"] is True
    assert "confirms OS dispatch only" in result["message"]

    assert len(launched_commands) == 1
    assert launched_commands[0] == ["/usr/bin/open", "https://google.com"]


def test_browser_controller_dispatch_missing_binary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)

    controller = BrowserController(open_binary="/missing/open")
    result = controller.open_website("https://example.com")

    assert result["success"] is False
    assert "was not found" in result["error"]


def test_browser_controller_dispatch_process_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)

    def failing_launcher(cmd: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    controller = BrowserController(
        open_binary="/usr/bin/open",
        launcher=failing_launcher,
    )

    result = controller.open_website("https://example.com")
    assert result["success"] is False
    assert "exit code 1" in result["error"]


def test_browser_controller_dispatch_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)

    def timeout_launcher(cmd: list[str], **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

    controller = BrowserController(
        open_binary="/usr/bin/open",
        launcher=timeout_launcher,
    )

    result = controller.open_website("https://example.com")
    assert result["success"] is False
    assert "timed out after 5.0 seconds" in result["error"]
