"""Small DeepSeek JSON client that never persists or logs the API key."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DeepSeekAPIError(RuntimeError):
    """Raised when the remote request or response contract fails."""


class DeepSeekResponseParseError(DeepSeekAPIError):
    """JSON parsing failure that preserves the paid response for audit."""

    def __init__(self, message: str, *, raw_content: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.metadata = metadata


class DeepSeekJSONClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: int = 180,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Set DEEPSEEK_API_KEY in the process environment")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        reasoning_effort: str = "high",
        thinking_enabled: bool = True,
        retries: int = 2,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking_enabled:
            payload["reasoning_effort"] = reasoning_effort
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                message = envelope["choices"][0]["message"]
                content = message.get("content")
                if not content:
                    safe_details = {
                        "finish_reason": envelope["choices"][0].get("finish_reason"),
                        "message_fields": sorted(message),
                        "usage": envelope.get("usage", {}),
                    }
                    raise DeepSeekAPIError(
                        "DeepSeek returned no visible message content: "
                        + json.dumps(safe_details, ensure_ascii=False)
                    )
                metadata = {
                    "model": envelope.get("model", self.model),
                    "usage": envelope.get("usage", {}),
                    "finish_reason": envelope["choices"][0].get("finish_reason"),
                }
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise DeepSeekResponseParseError(
                        f"DeepSeek content is not valid JSON: {exc}",
                        raw_content=content,
                        metadata=metadata,
                    ) from exc
                if not isinstance(parsed, dict):
                    raise DeepSeekAPIError("DeepSeek JSON content must be one object")
                return parsed, metadata
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == retries:
                    raise DeepSeekAPIError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == retries:
                    raise DeepSeekAPIError(f"DeepSeek request failed: {exc}") from exc
            time.sleep(2**attempt)
        raise AssertionError("unreachable")
