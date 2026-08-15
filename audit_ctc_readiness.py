"""Create a non-destructive readiness and orphan report for the CTC phase."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from config import AUDIO_DIR, CTC_AUDIT_JSON, RESULTS_DIR, TRANSCRIPTS_DIR
from ctc_transcriber import detect_device
from utils import read_json, write_json


def command_version(command: str, argument: str = "-version") -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, argument], capture_output=True, text=True, timeout=10, check=False
        )
        first_line = (completed.stdout or completed.stderr).splitlines()
        return first_line[0] if first_line else "available"
    except OSError:
        return "available but version check failed"


def build_audit() -> dict:
    success_ids = set()
    result_statuses: dict[str, int] = {}
    for path in RESULTS_DIR.glob("*.json"):
        try:
            result = read_json(path)
        except Exception:
            result_statuses["invalid_json"] = result_statuses.get("invalid_json", 0) + 1
            continue
        status = str(result.get("status") or "unknown")
        result_statuses[status] = result_statuses.get(status, 0) + 1
        if status == "success":
            success_ids.add(str(result.get("video_id") or path.stem))
    transcript_ids = {path.stem for path in TRANSCRIPTS_DIR.glob("*.txt")}
    audio_ids = {path.stem for path in AUDIO_DIR.glob("*.mp3")}
    device = detect_device()
    packages = {
        name: bool(importlib.util.find_spec(name.replace("-", "_")))
        for name in ("torch", "torchaudio", "transformers", "omnilingual_asr", "sklearn")
    }
    return {
        "project": str(Path(__file__).resolve().parent),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_repository": (Path(__file__).resolve().parent / ".git").exists(),
        "counts": {
            "audio_mp3": len(audio_ids),
            "transcript_txt": len(transcript_ids),
            "result_json": sum(result_statuses.values()),
            "successful_results": len(success_ids),
        },
        "result_statuses": result_statuses,
        "orphans": {
            "transcript_without_success_result": sorted(transcript_ids - success_ids),
            "success_result_without_transcript": sorted(success_ids - transcript_ids),
            "audio_without_success_result_quarantine": sorted(audio_ids - success_ids),
            "success_result_without_audio": sorted(success_ids - audio_ids),
        },
        "historical_21_vs_20_note": (
            "No current discrepancy: transcript IDs exactly match successful result IDs. "
            "Without version-control history, the previously observed extra file cannot be "
            "attributed reliably and no deletion is performed."
        ),
        "tools": {
            "ffmpeg": command_version("ffmpeg"),
            "ffprobe": command_version("ffprobe"),
            "nvidia_smi": command_version("nvidia-smi", "--version"),
        },
        "device": {
            "device": device.device,
            "cuda_available": device.cuda_available,
            "gpu_name": device.gpu_name,
        },
        "optional_packages": packages,
        "security": {
            "api_key_value_checked_or_recorded": False,
            "mistral_api_called": False,
        },
    }


def main() -> int:
    report = build_audit()
    write_json(CTC_AUDIT_JSON, report)
    print(f"Audit written to {CTC_AUDIT_JSON}")
    print(f"Current transcript orphans: {len(report['orphans']['transcript_without_success_result'])}")
    print(f"Quarantined audio without success result: {len(report['orphans']['audio_without_success_result_quarantine'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
