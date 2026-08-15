"""yt-dlp metadata and audio-download helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def check_yt_dlp() -> tuple[bool, str]:
    try:
        import yt_dlp  # noqa: F401
        return True, "yt-dlp Python package is available."
    except ImportError:
        return False, "yt-dlp is missing. Run: pip install -r requirements.txt"


def check_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True, "FFmpeg and ffprobe are available."
    return (
        False,
        "FFmpeg/ffprobe not found. In Windows PowerShell run: "
        "winget install --id Gyan.FFmpeg -e, then open a new PowerShell.",
    )


def get_video_metadata(url: str) -> dict[str, Any]:
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration")
    return {
        "video_id": str(info.get("id") or ""),
        "title": str(info.get("title") or ""),
        "duration": float(duration) if duration is not None else None,
        "youtube_url": str(info.get("webpage_url") or url),
    }


def download_audio(url: str, video_id: str, work_dir: Path) -> Path:
    """Download one source audio file without using its title as a filename."""
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(work_dir / f"{video_id}.source.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 60,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise RuntimeError("yt-dlp did not produce a non-empty audio file.")
    return downloaded
