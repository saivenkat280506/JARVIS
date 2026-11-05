"""Browser automation tool for securely opening validated websites on macOS."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_DISALLOWED_CHARACTERS = set(";`$&|<>\"' \t\r\n\\")

_FORBIDDEN_SCHEMES = {
    "file",
    "javascript",
    "data",
    "vbscript",
    "about",
    "blob",
    "ftp",
    "ssh",
    "telnet",
    "mailto",
    "ws",
    "wss",
    "intent",
}

_MAX_URL_LENGTH = 2048


def normalize_and_validate_url(raw_url: str) -> str:
    """Validate and normalize a user/model provided URL.

    - Rejects empty, overlong, or malicious URLs.
    - Prohibits non-web schemes (file, javascript, data, etc.).
    - Prohibits shell injection characters and embedded user/password credentials.
    - Normalizes bare domains (e.g. 'google.com') to 'https://google.com'.
    """
    clean = raw_url.strip()
    if not clean:
        raise ValueError("URL cannot be empty.")
    if len(clean) > _MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length of {_MAX_URL_LENGTH} characters.")

    if any(ch in _DISALLOWED_CHARACTERS for ch in clean):
        raise ValueError("URL contains disallowed syntax or shell control characters.")

    # Check for forbidden scheme prefixes
    if ":" in clean:
        prefix = clean.split(":", 1)[0].lower()
        if prefix in _FORBIDDEN_SCHEMES:
            raise ValueError(
                f"Scheme '{prefix}' is forbidden. Only HTTP and HTTPS are permitted."
            )

    # Prepend https:// if no scheme is provided
    if not (clean.startswith("http://") or clean.startswith("https://")):
        clean = "https://" + clean

    parsed = urlsplit(clean)

    # Enforce allowed schemes
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(
            f"Scheme '{parsed.scheme}' is not permitted. Only HTTP and HTTPS are allowed."
        )

    # Disallow embedded credentials (user:pass@host)
    if parsed.username or parsed.password:
        raise ValueError("Embedded credentials in URLs are strictly forbidden.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid network hostname.")

    # Validate hostname syntax
    if not re.match(r"^[a-zA-Z0-9.-]+$", hostname):
        raise ValueError(f"Invalid characters in hostname '{hostname}'.")

    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc,
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))


class BrowserController:
    """Safe controller dispatching URLs to macOS default browser."""

    def __init__(
        self,
        open_binary: str = "/usr/bin/open",
        launcher: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.open_binary = open_binary
        self._launcher = launcher or subprocess.run

    def open_website(self, url: str) -> Dict[str, Any]:
        """Validate and open the target website in macOS default browser."""
        try:
            normalized_url = normalize_and_validate_url(url)
        except ValueError as exc:
            return {
                "success": False,
                "error": f"Invalid or unsafe URL: {exc}",
            }

        if not os.path.exists(self.open_binary):
            return {
                "success": False,
                "error": f"macOS browser launcher binary '{self.open_binary}' was not found.",
            }

        try:
            # Fixed arguments, shell=False to prevent shell injection
            self._launcher(
                [self.open_binary, normalized_url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
            return {
                "success": True,
                "url": normalized_url,
                "dispatched": True,
                "message": (
                    f"Successfully dispatched '{normalized_url}' to macOS default browser. "
                    "Note: This confirms OS dispatch only, not whether the destination server resolved or rendered."
                ),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Browser dispatch process timed out after 5.0 seconds.",
            }
        except subprocess.CalledProcessError as exc:
            return {
                "success": False,
                "error": f"Browser launch failed with exit code {exc.returncode}.",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to dispatch browser: {exc}",
            }


# Global shared instance
default_browser_controller = BrowserController()


def open_website(url: str) -> Dict[str, Any]:
    """Tool handler for open_website."""
    return default_browser_controller.open_website(url)
