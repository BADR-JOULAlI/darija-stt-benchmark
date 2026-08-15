"""Official Mistral Speech-to-Text client wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    pass


def _friendly_http_error(status: int | None) -> str:
    messages = {
        400: "Bad request: check the audio file and model name.",
        401: "Authentication failed: check MISTRAL_API_KEY.",
        403: "Access forbidden: the key may not have access to this model.",
        404: "Endpoint or model not found: check MISTRAL_MODEL.",
        413: "Audio file is too large for the API.",
        429: "Rate limit reached: wait and retry later.",
    }
    if status in messages:
        return messages[status]
    if status is not None and status >= 500:
        return f"Mistral service error ({status}); retry later."
    return "Mistral API request failed."


def transcribe_audio(audio_path: str | Path) -> tuple[str, dict[str, Any]]:
    path = Path(audio_path)
    if not path.is_file():
        raise TranscriptionError(f"Audio file not found: {path}")
    if path.stat().st_size <= 0:
        raise TranscriptionError(f"Audio file is empty: {path}")

    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_MODEL")
    if not api_key:
        raise TranscriptionError("MISTRAL_API_KEY is not configured.")
    if not model:
        raise TranscriptionError("MISTRAL_MODEL is not configured.")

    try:
        from mistralai.client import Mistral, errors

        with Mistral(api_key=api_key) as client, path.open("rb") as audio_file:
            response = client.audio.transcriptions.complete(
                model=model,
                file={"content": audio_file, "file_name": path.name},
                diarize=False,
                timeout_ms=120_000,
            )
        text = str(response.text or "").strip()
        metadata = {
            "model": str(getattr(response, "model", model) or model),
            "language": getattr(response, "language", None),
            "usage": (
                response.usage.model_dump(mode="json")
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            ),
        }
        return text, metadata
    except ImportError as exc:
        raise TranscriptionError("mistralai is missing. Run: pip install -r requirements.txt") from exc
    except errors.MistralError as exc:
        raise TranscriptionError(_friendly_http_error(getattr(exc, "status_code", None))) from exc
    except TimeoutError as exc:
        raise TranscriptionError("Mistral request timed out.") from exc
    except OSError as exc:
        raise TranscriptionError(f"Network or file error: {exc}") from exc
    except Exception as exc:
        # Do not include request headers or objects that could contain credentials.
        raise TranscriptionError(f"Mistral transcription failed: {type(exc).__name__}") from exc
