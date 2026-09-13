"""Verify the cleaned MoulSot transcription CSV hosted on Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REQUIRED_COLUMNS = {
    "id", "original_text", "new_transcription", "training_text",
    "training_ready", "review_status", "latin_code_switch_present",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes"}
    )


def verify(repo_id: str, filename: str, token: str | None) -> dict:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=token,
    )
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    text = frame["training_text"].fillna("").astype(str).str.strip()
    original = frame["original_text"].fillna("").astype(str).str.strip()
    ready = as_bool(frame["training_ready"].fillna(False))
    duration = pd.to_numeric(frame["duration"], errors="coerce")
    changed = original != text
    report = {
        "repo_id": repo_id,
        "source_file": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_required_columns": missing,
        "unique_ids": int(frame["id"].nunique(dropna=True)),
        "duplicate_ids": int(frame["id"].duplicated(keep=False).sum()),
        "training_ready_rows": int(ready.sum()),
        "not_training_ready_rows": int((~ready).sum()),
        "empty_training_text_rows": int((text == "").sum()),
        "empty_original_text_rows": int((original == "").sum()),
        "changed_training_text_rows": int(changed.sum()),
        "changed_and_ready_rows": int((changed & ready).sum()),
        "code_switch_rows": int(as_bool(frame["latin_code_switch_present"].fillna(False)).sum()),
        "code_switch_and_ready_rows": int((as_bool(frame["latin_code_switch_present"].fillna(False)) & ready).sum()),
        "duration_invalid_rows": int(duration.isna().sum()),
        "duration_zero_or_negative_rows": int((duration <= 0).fillna(False).sum()),
        "duration_seconds": {
            "total": float(duration.sum()),
            "min": float(duration.min()),
            "median": float(duration.median()),
            "max": float(duration.max()),
        },
        "review_status_counts": {
            str(key): int(value) for key, value in frame["review_status"].fillna("<NA>").value_counts().items()
        },
        "examples_changed_ready": frame.loc[changed & ready, [
            "id", "original_text", "new_transcription", "training_text", "review_status"
        ]].head(20).fillna("").to_dict(orient="records"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="sailu4/lmaanaDataset")
    parser.add_argument("--filename", default="moulsot_cleaned_for_training.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/moulsot_cleaning_verification.json"))
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN")
    report = verify(args.repo_id, args.filename, token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
