"""Resumable Mistral transcription pilot for a local MoulSot audio sample."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from transcriber import TranscriptionError, transcribe_audio


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def audio_files(root: Path, limit: int) -> list[Path]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files if limit <= 0 else files[:limit]


def run(root: Path, output: Path, limit: int, force: bool = False) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    transcripts = output / "transcripts"
    results_dir = output / "results"
    transcripts.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    rows = []
    files = audio_files(root, limit)
    for index, audio_path in enumerate(files, 1):
        sample_id = audio_path.stem
        result_path = results_dir / f"{sample_id}.json"
        transcript_path = transcripts / f"{sample_id}.txt"
        if result_path.exists() and not force:
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            print(f"[{index}/{len(files)}] SKIP {sample_id}: already processed")
            continue

        started = time.perf_counter()
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "status": "error",
            "transcription": "",
            "error": "",
        }
        try:
            text, metadata = transcribe_audio(audio_path)
            transcript_path.write_text(text, encoding="utf-8")
            result.update(
                status="success",
                transcription=text,
                model=metadata.get("model"),
                language=metadata.get("language"),
                usage=metadata.get("usage"),
            )
            print(f"[{index}/{len(files)}] OK {sample_id}")
        except TranscriptionError as exc:
            result["error"] = str(exc)
            print(f"[{index}/{len(files)}] ERROR {sample_id}: {exc}")
        result["processing_time_seconds"] = round(time.perf_counter() - started, 3)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(result)

    (output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/moulsot_mistral_200"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY is not configured in this Jupyter session.")
    if not os.getenv("MISTRAL_MODEL"):
        raise SystemExit("MISTRAL_MODEL is not configured in this Jupyter session.")
    if not args.audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {args.audio_dir}")
    rows = run(args.audio_dir, args.output_root, args.limit, args.force)
    print(f"Completed: {sum(row.get('status') == 'success' for row in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
