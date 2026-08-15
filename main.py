"""Optional CLI equivalent of the notebook workflow."""

from __future__ import annotations

import argparse
import os

from benchmark import prepare_samples, transcribe_samples
from config import DEFAULT_SAMPLE_COUNT
from dataset_loader import inspect_dataset
from youtube_downloader import check_ffmpeg, check_yt_dlp


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Mistral STT on existing Darija YouTube URLs.")
    parser.add_argument("--dataset", required=True, help="Existing SQLite/CSV/JSON/JSONL/TXT dataset")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    info = inspect_dataset(args.dataset)
    print(f"Dataset: {info['path']}")
    print(f"URL source: table={info['table']!r}, column={info['url_column']!r}")
    if info.get("filter"):
        print(f"Dataset filter: {info['filter']}")
    for check in (check_yt_dlp, check_ffmpeg):
        ok, message = check()
        print(message)
        if not ok:
            return 2
    if not os.getenv("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY is not configured.")
        return 2
    if not os.getenv("MISTRAL_MODEL"):
        print("MISTRAL_MODEL is not configured.")
        return 2

    selected = prepare_samples(args.dataset, args.count)
    transcribe_samples(selected, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
