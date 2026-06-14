import os
import tempfile
from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel

WHISPER_MODEL_OPTIONS = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large-v3",
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3-turbo",
]

_model: WhisperModel | None = None
_model_name: str | None = None
_model_lock = Lock()


def set_whisper_model(model_name: str) -> None:
    global _model, _model_name
    with _model_lock:
        if _model_name != model_name:
            _model_name = model_name
            _model = None


def get_whisper_model() -> WhisperModel:
    global _model, _model_name
    target_name = _model_name or os.getenv("WHISPER_MODEL", "base.en")
    if _model is None or _model_name != target_name:
        with _model_lock:
            if _model is None or _model_name != target_name:
                device = os.getenv("WHISPER_DEVICE", "cpu")
                compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
                _model = WhisperModel(target_name, device=device, compute_type=compute_type)
                _model_name = target_name
    return _model


def transcribe_audio_bytes(data: bytes, filename: str = "audio.webm") -> str:
    suffix = Path(filename).suffix or ".webm"
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        model = get_whisper_model()
        transcribe_kwargs: dict = {
            "beam_size": 1,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        model_name = _model_name or os.getenv("WHISPER_MODEL", "base.en")
        if model_name.endswith(".en"):
            transcribe_kwargs["language"] = "en"
        segments, _ = model.transcribe(path, **transcribe_kwargs)
        return "".join(segment.text for segment in segments).strip()
    finally:
        if path:
            os.unlink(path)
