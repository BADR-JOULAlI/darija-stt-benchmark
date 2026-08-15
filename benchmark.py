"""Notebook-friendly orchestration for preparation, transcription and summaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from audio_extractor import extract_audio_sample, middle_sample_times
from config import (
    AUDIO_DIR, DEFAULT_SAMPLE_COUNT, RESULTS_DIR, SAMPLE_DURATION_SECONDS,
    SELECTED_VIDEOS_CSV, SUMMARY_CSV, SUMMARY_JSON, TRANSCRIPTS_DIR,
    ensure_directories,
)
from dataset_loader import iter_youtube_records
from transcriber import TranscriptionError, transcribe_audio
from utils import read_json, write_csv, write_json
from youtube_downloader import download_audio, get_video_metadata


SELECTION_FIELDS = [
    "index", "video_id", "youtube_url", "title", "original_duration_seconds",
    "sample_start_seconds", "sample_end_seconds", "sample_duration_seconds", "status", "error",
]
SUMMARY_FIELDS = [
    "index", "video_id", "title", "youtube_url", "original_duration_seconds",
    "sample_start_seconds", "sample_end_seconds", "sample_duration_seconds", "status",
    "processing_time_seconds", "transcript_file", "error",
]


def prepare_samples(dataset_path: str | Path, target_count: int = DEFAULT_SAMPLE_COUNT) -> list[dict[str, Any]]:
    ensure_directories()
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for source in iter_youtube_records(dataset_path):
        if len(selected) >= target_count:
            break
        url = source["youtube_url"]
        display = len(selected) + 1
        print(f"[{display}/{target_count}] Checking video metadata...")
        row: dict[str, Any] = {
            "index": len(attempts) + 1,
            "video_id": "",
            "youtube_url": url,
            "title": "",
            "original_duration_seconds": "",
            "sample_start_seconds": "",
            "sample_end_seconds": "",
            "sample_duration_seconds": SAMPLE_DURATION_SECONDS,
            "status": "unavailable",
            "error": "",
        }
        try:
            metadata = get_video_metadata(url)
            row.update(
                video_id=metadata["video_id"], title=metadata["title"],
                youtube_url=metadata["youtube_url"],
                original_duration_seconds=metadata["duration"],
            )
            if not metadata["duration"] or metadata["duration"] < SAMPLE_DURATION_SECONDS:
                row["status"] = "too_short"
                print(f"[{display}/{target_count}] SKIP: too short")
                attempts.append(row)
                continue

            start, end = middle_sample_times(metadata["duration"], SAMPLE_DURATION_SECONDS)
            row.update(sample_start_seconds=start, sample_end_seconds=end)
            print(f"[{display}/{target_count}] Extracting 120-second sample...")
            source_audio = download_audio(row["youtube_url"], row["video_id"], AUDIO_DIR)
            output_audio = AUDIO_DIR / f"{row['video_id']}.mp3"
            extract_audio_sample(source_audio, output_audio, start, SAMPLE_DURATION_SECONDS)
            source_audio.unlink(missing_ok=True)
            row["status"] = "selected"
            selected.append(row)
            print(f"[{display}/{target_count}] SAMPLE READY")
        except Exception as exc:
            row["status"] = "download_error" if row["video_id"] else "unavailable"
            row["error"] = str(exc)[:500]
            print(f"[{display}/{target_count}] SKIP: {row['status']}")
        attempts.append(row)
        write_csv(SELECTED_VIDEOS_CSV, attempts, SELECTION_FIELDS)

    write_csv(SELECTED_VIDEOS_CSV, attempts, SELECTION_FIELDS)
    print(f"Prepared {len(selected)} valid samples from {len(attempts)} attempted URLs.")
    return selected


def transcribe_samples(selected: list[dict[str, Any]], force: bool = False) -> list[dict[str, Any]]:
    ensure_directories()
    summary: list[dict[str, Any]] = []
    total = len(selected)
    for position, item in enumerate(selected, start=1):
        video_id = str(item["video_id"])
        result_path = RESULTS_DIR / f"{video_id}.json"
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
        if result_path.exists() and not force:
            existing = read_json(result_path)
            if existing.get("status") == "success":
                print(f"Skipping {video_id}: already successfully processed.")
                summary.append(_summary_row(item, existing, transcript_path))
                continue

        print(f"[{position}/{total}] Transcribing...")
        started = time.perf_counter()
        result = {**item, "model": None, "status": "error", "transcription": ""}
        try:
            text, metadata = transcribe_audio(AUDIO_DIR / f"{video_id}.mp3")
            elapsed = time.perf_counter() - started
            transcript_path.write_text(text, encoding="utf-8")
            result.update(
                model=metadata["model"], language=metadata.get("language"),
                usage=metadata.get("usage"), status="success", transcription=text,
                processing_time_seconds=elapsed,
            )
            print(f"[{position}/{total}] SUCCESS")
        except TranscriptionError as exc:
            elapsed = time.perf_counter() - started
            result.update(processing_time_seconds=elapsed, error=str(exc))
            print(f"[{position}/{total}] ERROR: {exc}")
        write_json(result_path, result)
        summary.append(_summary_row(item, result, transcript_path))
        _write_summary(summary)
    _write_summary(summary)
    print_terminal_summary(summary, requested=total)
    return summary


def _summary_row(item: dict[str, Any], result: dict[str, Any], transcript_path: Path) -> dict[str, Any]:
    return {
        **{key: item.get(key, "") for key in SUMMARY_FIELDS},
        "status": result.get("status", "error"),
        "processing_time_seconds": result.get("processing_time_seconds", 0),
        "transcript_file": str(transcript_path) if result.get("status") == "success" else "",
        "error": result.get("error", ""),
    }


def _write_summary(summary: list[dict[str, Any]]) -> None:
    write_csv(SUMMARY_CSV, summary, SUMMARY_FIELDS)
    write_json(SUMMARY_JSON, summary)


def print_terminal_summary(summary: list[dict[str, Any]], requested: int) -> None:
    successful = sum(row["status"] == "success" for row in summary)
    failed = len(summary) - successful
    times = [float(row.get("processing_time_seconds") or 0) for row in summary]
    total_time = sum(times)
    selection_attempts: list[dict[str, Any]] = []
    if SELECTED_VIDEOS_CSV.exists():
        import csv
        with SELECTED_VIDEOS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            selection_attempts = list(csv.DictReader(handle))
    too_short = sum(row.get("status") == "too_short" for row in selection_attempts)
    unavailable = sum(row.get("status") == "unavailable" for row in selection_attempts)
    download_errors = sum(row.get("status") == "download_error" for row in selection_attempts)
    print("\n====================================")
    print("BENCHMARK COMPLETE")
    print("====================================")
    print(f"Requested valid samples: {requested}")
    print(f"Successful transcriptions: {successful}")
    print(f"Failed transcriptions: {failed}")
    print(f"Too short videos: {too_short}")
    print(f"Unavailable videos: {unavailable}")
    print(f"Download errors: {download_errors}")
    print(f"Total audio duration: {successful * SAMPLE_DURATION_SECONDS:g} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time: {(total_time / successful if successful else 0):.2f} seconds")
    print(f"Results: {SUMMARY_CSV}")
