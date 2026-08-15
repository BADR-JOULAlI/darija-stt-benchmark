import builtins
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ctc_transcriber
from calibrate_threshold import CalibrationError, load_annotated_rows, split_by_video_id
from ctc_transcriber import (
    CTCError, CTCOutput, DeviceInfo, prepare_annotation_template, run_ctc_pipeline,
)
from quality_scorer import cer, compare_transcripts, wer
from text_normalizer import NormalizationConfig, normalize_text
from utils import read_json, write_json


class FakeBackend:
    def __init__(self, text="سلام عليكم", fail=False):
        self.text = text
        self.fail = fail
        self.calls = 0

    def transcribe_batch(self, paths, batch_size):
        self.calls += 1
        if self.fail:
            raise CTCError("synthetic load or inference error")
        return [CTCOutput(self.text, confidence=0.8, blank_rate=0.4) for _ in paths]


class TextAndMetricsTests(unittest.TestCase):
    def test_conservative_normalization_preserves_code_switching(self):
        self.assertEqual(normalize_text("  Salam   خويا Café  "), "salam خويا café")

    def test_configurable_arabic_normalization_and_digits(self):
        cfg = NormalizationConfig(
            remove_punctuation=True,
            remove_arabic_diacritics=True,
            normalize_alef=True,
            normalize_ya=True,
            normalize_digits=True,
        )
        self.assertEqual(normalize_text("إِلَى ٢٠٢٦!", cfg), "الي 2026")
        self.assertEqual(normalize_text("سلام، كيداير؟", cfg), "سلام كيداير")

    def test_arabizi_is_not_destroyed(self):
        self.assertEqual(normalize_text("3lach Ma-jitich?"), "3lach ma-jitich?")

    def test_cer_wer_and_empty_results(self):
        self.assertAlmostEqual(cer("abc", "adc"), 1 / 3)
        self.assertAlmostEqual(wer("salam labas", "salam"), 0.5)
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(wer("", "extra"), 1.0)
        metrics = compare_transcripts("", "")
        self.assertEqual(metrics["character_similarity"], 1.0)
        self.assertEqual(metrics["provisional_quality_score"], 0.0)
        self.assertFalse(metrics["score_is_calibrated"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.results = self.root / "results"
        self.ctc = self.root / "ctc_results"
        self.quality = self.root / "quality_results"
        self.annotation = self.root / "annotations.csv"
        for directory in (self.audio, self.results, self.ctc, self.quality):
            directory.mkdir()
        self.patchers = [
            patch.object(ctc_transcriber, "AUDIO_DIR", self.audio),
            patch.object(ctc_transcriber, "RESULTS_DIR", self.results),
            patch.object(ctc_transcriber, "CTC_RESULTS_DIR", self.ctc),
            patch.object(ctc_transcriber, "QUALITY_RESULTS_DIR", self.quality),
            patch.object(ctc_transcriber, "ANNOTATION_CSV", self.annotation),
            patch.object(ctc_transcriber, "ensure_directories", lambda: None),
            patch.object(ctc_transcriber, "detect_device", lambda: DeviceInfo("cpu", False, None)),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def add_mistral_success(self, video_id="sample"):
        write_json(self.results / f"{video_id}.json", {
            "video_id": video_id,
            "status": "success",
            "model": "darija-stt-solutions-2607",
            "transcription": "سلام عليكم",
        })

    def test_audio_absent_is_recorded_without_backend_call(self):
        self.add_mistral_success()
        backend = FakeBackend()
        results = run_ctc_pipeline(limit=1, backend=backend)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("Audio file not found", results[0]["error"])
        self.assertEqual(backend.calls, 0)

    def test_annotation_template_exists_before_ctc(self):
        self.add_mistral_success()
        self.assertEqual(prepare_annotation_template(), 1)
        with self.annotation.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["segment_id"], "sample")
        self.assertEqual(row["ctc_transcript"], "")
        self.assertEqual(row["human_label"], "")

    def test_success_is_written_and_cached_on_resume(self):
        self.add_mistral_success()
        (self.audio / "sample.mp3").write_bytes(b"not-decoded-by-fake-backend")
        backend = FakeBackend()
        first = run_ctc_pipeline(limit=1, backend=backend)
        self.assertEqual(first[0]["status"], "success")
        self.assertEqual(backend.calls, 1)
        second_backend = FakeBackend(fail=True)
        second = run_ctc_pipeline(limit=1, backend=second_backend)
        self.assertEqual(second[0]["status"], "success")
        self.assertEqual(second_backend.calls, 0)
        self.assertTrue((self.quality / "sample.json").is_file())
        with self.annotation.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["human_label"], "")

    def test_backend_error_is_safe_and_resumable(self):
        self.add_mistral_success()
        (self.audio / "sample.mp3").write_bytes(b"x")
        result = run_ctc_pipeline(limit=1, backend=FakeBackend(fail=True))[0]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "synthetic load or inference error")
        saved = read_json(self.ctc / "sample.json")
        self.assertEqual(saved["status"], "error")

    def test_no_gpu_detection_without_torch(self):
        original_import = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("synthetic")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=importing):
            device = ctc_transcriber.detect_device()
        self.assertFalse(device.cuda_available)
        self.assertEqual(device.device, "cpu")


class CalibrationTests(unittest.TestCase):
    def test_refuses_absent_human_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_id", "human_label"])
                writer.writeheader()
                writer.writerow({"video_id": "a", "human_label": ""})
            with self.assertRaises(CalibrationError):
                load_annotated_rows(path, minimum=1)

    def test_video_groups_do_not_overlap(self):
        rows = [
            {"video_id": video, "human_label": "good"}
            for video in ("a", "a", "b", "c", "d", "e")
        ]
        calibration, test = split_by_video_id(rows, seed=7)
        self.assertFalse(
            {row["video_id"] for row in calibration}
            & {row["video_id"] for row in test}
        )


if __name__ == "__main__":
    unittest.main()
