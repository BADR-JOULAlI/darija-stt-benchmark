"""Calibrate a high-precision 'good' decision after human annotation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from config import ANNOTATION_CSV, OUTPUTS_DIR
from utils import write_json


ALLOWED_LABELS = {"good", "medium", "reject"}
FEATURES = [
    "character_similarity", "word_similarity", "length_ratio", "ctc_confidence",
]
DEFAULT_REPORT = OUTPUTS_DIR / "ctc_calibration_report.json"


class CalibrationError(RuntimeError):
    pass


def load_annotated_rows(path: str | Path, minimum: int = 30) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    annotated = []
    for row in rows:
        label = str(row.get("human_label") or "").strip().lower()
        if not label:
            continue
        if label not in ALLOWED_LABELS:
            raise CalibrationError(f"Invalid human_label {label!r} for {row.get('segment_id')!r}.")
        if not row.get("video_id"):
            raise CalibrationError("Every annotated row must have a video_id.")
        parsed = dict(row)
        parsed["human_label"] = label
        for feature in FEATURES + ["provisional_quality_score"]:
            value = row.get(feature)
            parsed[feature] = float(value) if value not in (None, "") else None
        if parsed["provisional_quality_score"] is None:
            parsed["provisional_quality_score"] = provisional_score(parsed)
        annotated.append(parsed)
    if len(annotated) < minimum:
        raise CalibrationError(
            f"At least {minimum} human-annotated rows are required; found {len(annotated)}."
        )
    groups = {row["video_id"] for row in annotated}
    if len(groups) < 4:
        raise CalibrationError("At least four distinct video_id groups are required.")
    labels = Counter(row["human_label"] for row in annotated)
    if labels["good"] < 5 or len(annotated) - labels["good"] < 5:
        raise CalibrationError("Need at least five good and five non-good examples.")
    return annotated


def provisional_score(row: dict[str, Any]) -> float:
    values = []
    for name, weight in (("character_similarity", 0.4), ("word_similarity", 0.4), ("length_ratio", 0.2)):
        if row.get(name) is not None:
            values.append((float(row[name]), weight))
    if row.get("ctc_confidence") is not None:
        values.append((float(row["ctc_confidence"]), 0.2))
    if not values:
        raise CalibrationError("No usable numeric quality features were found.")
    return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)


def split_by_video_id(
    rows: list[dict[str, Any]], test_fraction: float = 0.3, seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = sorted({str(row["video_id"]) for row in rows})
    random.Random(seed).shuffle(groups)
    test_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * test_fraction)))
    test_groups = set(groups[:test_count])
    calibration = [row for row in rows if str(row["video_id"]) not in test_groups]
    test = [row for row in rows if str(row["video_id"]) in test_groups]
    if not calibration or not test:
        raise CalibrationError("Group split produced an empty partition.")
    return calibration, test


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision_good": precision,
        "recall_good": recall,
        "f1_good": f1,
        "retention_rate": sum(y_pred) / len(y_pred) if y_pred else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_threshold(scores: list[float], labels: list[int], target_precision: float = 0.9) -> float:
    candidates = sorted(set(scores))
    evaluated = []
    for threshold in candidates:
        metrics = binary_metrics(labels, [int(score >= threshold) for score in scores])
        evaluated.append((threshold, metrics))
    eligible = [item for item in evaluated if item[1]["precision_good"] >= target_precision]
    if eligible:
        return max(eligible, key=lambda item: (item[1]["recall_good"], item[1]["f1_good"]))[0]
    return max(evaluated, key=lambda item: item[1]["f1_good"])[0]


def feature_matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    matrix = []
    for row in rows:
        matrix.append([
            float(row.get("character_similarity") or 0.0),
            float(row.get("word_similarity") or 0.0),
            float(row.get("length_ratio") or 0.0),
            float(row.get("ctc_confidence") or 0.0),
        ])
    return matrix


def calibrate(rows: list[dict[str, Any]], target_precision: float = 0.9) -> dict[str, Any]:
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise CalibrationError("scikit-learn is required for calibration.") from exc
    calibration_rows, test_rows = split_by_video_id(rows)
    y_cal = [int(row["human_label"] == "good") for row in calibration_rows]
    y_test = [int(row["human_label"] == "good") for row in test_rows]
    if len(set(y_cal)) < 2 or len(set(y_test)) < 2:
        raise CalibrationError("Both group partitions must contain good and non-good labels.")

    cal_scores = [float(row["provisional_quality_score"]) for row in calibration_rows]
    test_scores = [float(row["provisional_quality_score"]) for row in test_rows]
    manual_threshold = select_threshold(cal_scores, y_cal, target_precision)
    manual_pred = [int(score >= manual_threshold) for score in test_scores]

    logistic = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    logistic.fit(feature_matrix(calibration_rows), y_cal)
    logistic_cal_probs = logistic.predict_proba(feature_matrix(calibration_rows))[:, 1].tolist()
    logistic_test_probs = logistic.predict_proba(feature_matrix(test_rows))[:, 1].tolist()
    logistic_threshold = select_threshold(logistic_cal_probs, y_cal, target_precision)

    methods: dict[str, Any] = {
        "manual_threshold": {
            "threshold": manual_threshold,
            "test_metrics": binary_metrics(y_test, manual_pred),
        },
        "logistic_regression": {
            "threshold": logistic_threshold,
            "test_metrics": binary_metrics(
                y_test, [int(score >= logistic_threshold) for score in logistic_test_probs]
            ),
            "feature_order": FEATURES,
            "coefficients": logistic.coef_[0].tolist(),
            "intercept": logistic.intercept_.tolist(),
        },
    }
    if len(calibration_rows) >= 50:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(cal_scores, y_cal)
        iso_cal = isotonic.predict(cal_scores).tolist()
        iso_test = isotonic.predict(test_scores).tolist()
        iso_threshold = select_threshold(iso_cal, y_cal, target_precision)
        methods["isotonic"] = {
            "threshold": iso_threshold,
            "test_metrics": binary_metrics(y_test, [int(score >= iso_threshold) for score in iso_test]),
        }
    else:
        methods["isotonic"] = {"status": "skipped", "reason": "Requires at least 50 calibration rows."}

    ranked = [
        (name, value) for name, value in methods.items() if "test_metrics" in value
    ]
    recommended_name, recommended = max(
        ranked,
        key=lambda item: (
            item[1]["test_metrics"]["precision_good"],
            item[1]["test_metrics"]["recall_good"],
        ),
    )
    return {
        "methodology": "Human labels; group split by video_id; good vs non-good.",
        "warning": "Twenty 120-second clips are a technical pilot, not production validation.",
        "target_precision_good": target_precision,
        "counts": {
            "all": len(rows), "calibration": len(calibration_rows), "test": len(test_rows),
            "unique_videos": len({row['video_id'] for row in rows}),
        },
        "group_overlap": sorted(
            {row["video_id"] for row in calibration_rows}
            & {row["video_id"] for row in test_rows}
        ),
        "methods": methods,
        "recommended": {"method": recommended_name, "threshold": recommended["threshold"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a human-validated good/non-good threshold.")
    parser.add_argument("--annotations", default=str(ANNOTATION_CSV))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--minimum", type=int, default=30)
    parser.add_argument("--target-precision", type=float, default=0.9)
    args = parser.parse_args()
    try:
        rows = load_annotated_rows(args.annotations, minimum=args.minimum)
        report = calibrate(rows, target_precision=args.target_precision)
    except (OSError, CalibrationError) as exc:
        print(f"Calibration not run: {exc}")
        return 2
    write_json(Path(args.output), report)
    print(json.dumps(report["recommended"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
