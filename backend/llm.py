import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _uses_default_temperature_only(model: str) -> bool:
    normalized = model.lower().replace("_", "-").replace(" ", "-")
    return "gpt-5" in normalized or "gpt5" in normalized


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url


def _post_json(url: str, payload: dict, headers: dict, provider_name: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{provider_name} request failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {provider_name} at {url}") from exc


def chat(
    *,
    provider: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    provider = provider.lower().strip().replace("_", "-")
    base_url = _normalize_base_url(base_url)

    if provider == "ollama":
        options = {}
        if temperature is not None:
            options["temperature"] = temperature

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if options:
            payload["options"] = options

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = _post_json(f"{base_url}/api/chat", payload, headers, "Ollama")
        return data.get("message", {}).get("content", "")

    if provider not in {"openai", "openai-compatible"}:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "openai" and not api_key:
        raise ValueError("OpenAI API models require an API key")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None and not _uses_default_temperature_only(model):
        payload["temperature"] = temperature

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = _post_json(f"{base_url}/chat/completions", payload, headers, "OpenAI-compatible API")
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")
