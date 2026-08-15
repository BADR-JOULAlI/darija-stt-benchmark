"""Extract and validate the exact middle 120 seconds with FFmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def middle_sample_times(duration: float, sample_duration: float = 120.0) -> tuple[float, float]:
    if duration < sample_duration:
        raise ValueError(f"Video duration ({duration}) is shorter than {sample_duration} seconds.")
    start = max(0.0, (duration - sample_duration) / 2.0)
    return start, start + sample_duration


def extract_audio_sample(
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float = 120.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_seconds:.3f}", "-i", str(source_path),
        "-t", f"{duration_seconds:.3f}", "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-b:a", "64k", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "Unknown FFmpeg error."
        raise RuntimeError(f"FFmpeg extraction failed: {message}")
    measured = probe_duration(output_path)
    if abs(measured - duration_seconds) > 0.5:
        raise RuntimeError(
            f"Extracted audio duration is {measured:.3f}s; expected approximately {duration_seconds:.3f}s."
        )
    return output_path


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    return float(data["format"]["duration"])
