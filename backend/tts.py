import json
import urllib.error
import urllib.request
from typing import List, Optional

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
APP_SPEECH_RATE_MIN = 0.5
APP_SPEECH_RATE_MAX = 2.5
ELEVENLABS_SPEED_MIN = 0.7
ELEVENLABS_SPEED_MAX = 1.2


def _clamp_app_speech_rate(rate: Optional[float]) -> float:
    if rate is None or not isinstance(rate, (int, float)):
        return 1.0
    return max(APP_SPEECH_RATE_MIN, min(APP_SPEECH_RATE_MAX, float(rate)))


def _map_speech_rate_to_elevenlabs_speed(rate: Optional[float]) -> float:
    """Map app speech rate (0.5–2.5×) to ElevenLabs speed (0.7–1.2×)."""
    app_rate = _clamp_app_speech_rate(rate)
    if app_rate <= 1.0:
        return ELEVENLABS_SPEED_MIN + (app_rate - APP_SPEECH_RATE_MIN) / (1.0 - APP_SPEECH_RATE_MIN) * (
            1.0 - ELEVENLABS_SPEED_MIN
        )
    return 1.0 + (app_rate - 1.0) / (APP_SPEECH_RATE_MAX - 1.0) * (ELEVENLABS_SPEED_MAX - 1.0)


def _request(
    *,
    method: str,
    url: str,
    api_key: str,
    payload: Optional[dict] = None,
    accept: str = "application/json",
) -> bytes:
    headers = {
        "xi-api-key": api_key,
        "Accept": accept,
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"ElevenLabs request failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach ElevenLabs API") from exc


def _label_value(labels: dict, key: str) -> Optional[str]:
    value = labels.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_voice_name(name: str) -> tuple[str, Optional[str]]:
    if " - " not in name:
        return name, None
    display_name, detail = [part.strip() for part in name.split(" - ", 1)]
    if not display_name:
        return name, None
    return display_name, detail or None


def _build_characteristics(
    *,
    category: Optional[str],
    labels: dict,
    description: Optional[str],
) -> Optional[str]:
    segments: List[str] = []
    if category:
        segments.append(f"Category: {category}")
    accent = _label_value(labels, "accent")
    if accent:
        segments.append(f"Accent: {accent}")
    language = _label_value(labels, "language")
    if language:
        segments.append(f"Language: {language}")
    use_case = _label_value(labels, "use_case")
    if use_case:
        segments.append(f"Use case: {use_case}")
    label_description = _label_value(labels, "description")
    if label_description:
        segments.append(label_description)
    elif description:
        segments.append(description)
    return ". ".join(segments) if segments else None


def list_elevenlabs_voices(api_key: str) -> List[dict]:
    raw = _request(method="GET", url=f"{ELEVENLABS_API_BASE}/voices", api_key=api_key)
    data = json.loads(raw.decode("utf-8"))
    voices = data.get("voices", [])
    result = []
    for voice in voices:
        voice_id = (voice.get("voice_id") or "").strip()
        name = (voice.get("name") or voice_id or "Unknown voice").strip()
        if not voice_id:
            continue
        category = (voice.get("category") or "").strip() or None
        category_label = category or "voice"
        labels = voice.get("labels") if isinstance(voice.get("labels"), dict) else {}
        description = (voice.get("description") or "").strip() or None
        display_name, name_detail = _split_voice_name(name)
        characteristics = _build_characteristics(
            category=category,
            labels=labels,
            description=description,
        )
        if name_detail:
            characteristics = f"{name_detail}. {characteristics}" if characteristics else name_detail
        result.append(
            {
                "voice_id": voice_id,
                "name": display_name,
                "gender": _label_value(labels, "gender"),
                "age": _label_value(labels, "age"),
                "characteristics": characteristics,
                "label": f"{name} ({category_label}) — ElevenLabs",
            }
        )
    result.sort(key=lambda item: item["label"].lower())
    return result


def synthesize_elevenlabs_speech(
    *,
    api_key: str,
    voice_id: str,
    text: str,
    speech_rate: Optional[float] = None,
) -> bytes:
    payload = {
        "text": text.strip(),
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "speed": _map_speech_rate_to_elevenlabs_speed(speech_rate),
        },
    }
    return _request(
        method="POST",
        url=f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
        api_key=api_key,
        payload=payload,
        accept="audio/mpeg",
    )
