"""Asynchronous HTTP client for the local Ollama API."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit
import ipaddress
import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol defining the minimal interface required by AgentRuntime."""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request with message history and optional tools."""
        ...


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when connecting to the local Ollama daemon fails."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama inference request times out."""


class OllamaResponseError(OllamaError):
    """Raised when the Ollama daemon returns an HTTP error or malformed payload."""


class OllamaClient:
    """Asynchronous client for local loopback Ollama instances."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "qwen3:4b",
        default_num_ctx: int = 2048,
        default_num_predict: int = 512,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = self._validate_loopback(base_url)
        self.default_model = default_model
        self.default_num_ctx = default_num_ctx
        self.default_num_predict = default_num_predict
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None


    @staticmethod
    def _validate_loopback(url: str) -> str:
        """Validate and sanitize that the Ollama host is restricted to loopback."""
        normalized = url.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme != "http":
            raise ValueError(f"Ollama scheme must be 'http' (got '{parsed.scheme}')")
        host = (parsed.hostname or "").lower()
        is_loopback = host == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError(
                f"Security violation: Ollama endpoint must be loopback (got '{host}')"
            )
        return normalized

    async def get_client(self) -> httpx.AsyncClient:
        """Get or lazily create the underlying httpx AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaClient":
        await self.get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def is_available(self) -> bool:
        """Check whether local Ollama daemon is reachable on loopback."""
        try:
            client = await self.get_client()
            resp = await client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the local Ollama instance."""
        target_model = model or self.default_model
        merged_options = {
            "num_ctx": self.default_num_ctx,
            "temperature": 0.0,
            "num_predict": self.default_num_predict,
            **(options or {}),
        }


        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": merged_options,
        }
        if tools:
            payload["tools"] = tools

        client = await self.get_client()
        endpoint = f"{self.base_url}/api/chat"

        try:
            response = await client.post(endpoint, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to local Ollama at {self.base_url}. Ensure 'ollama serve' is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama request to {endpoint} timed out after {self.timeout}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"HTTP transport failure: {exc}") from exc

        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaResponseError("Failed to parse JSON response from Ollama") from exc

        if not isinstance(data, dict) or "message" not in data:
            raise OllamaResponseError(
                "Malformed response from Ollama: missing 'message' field"
            )

        return data
