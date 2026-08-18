import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from audio_extractor import middle_sample_times
from benchmark import discover_existing_audio_samples
from config import ensure_directories
from dataset_loader import identify_url_column, inspect_dataset, iter_youtube_records


class CoreTests(unittest.TestCase):
    def test_middle_sample_times(self):
        self.assertEqual(middle_sample_times(600), (240.0, 360.0))
        self.assertEqual(middle_sample_times(120), (0.0, 120.0))
        with self.assertRaises(ValueError):
            middle_sample_times(119.9)

    def test_identifies_real_column_name(self):
        rows = [{"name": "x", "video_url": "https://www.youtube.com/watch?v=abc"}]
        self.assertEqual(identify_url_column(list(rows[0]), rows), "video_url")

    def test_read_only_sqlite_loader(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "videos.sqlite"
            with closing(sqlite3.connect(db)) as connection:
                connection.execute("CREATE TABLE videos (video_id TEXT, video_url TEXT, title TEXT)")
                connection.execute(
                    "INSERT INTO videos VALUES (?, ?, ?)",
                    ("abc", "https://www.youtube.com/watch?v=abc", "Test"),
                )
                connection.commit()
            info = inspect_dataset(db)
            self.assertEqual(info["table"], "videos")
            self.assertEqual(info["url_column"], "video_url")
            records = iter_youtube_records(db)
            try:
                self.assertTrue(next(records)["youtube_url"].endswith("abc"))
            finally:
                records.close()

    def test_directories_exist_after_setup(self):
        ensure_directories()

    def test_discovers_pending_existing_audio_without_dataset(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            results_dir = root / "results"
            audio_dir.mkdir()
            results_dir.mkdir()
            (audio_dir / "done.mp3").write_bytes(b"audio")
            (audio_dir / "pending.mp3").write_bytes(b"audio")
            (results_dir / "done.json").write_text(
                json.dumps({"status": "success"}), encoding="utf-8"
            )
            manifest = root / "selected.csv"
            manifest.write_text(
                "video_id,youtube_url,title\npending,https://youtu.be/p,Pending title\n",
                encoding="utf-8",
            )
            with (
                patch("benchmark.AUDIO_DIR", audio_dir),
                patch("benchmark.RESULTS_DIR", results_dir),
                patch("benchmark.SELECTED_VIDEOS_CSV", manifest),
                patch("benchmark.ensure_directories"),
            ):
                samples = discover_existing_audio_samples(pending_only=True)
            self.assertEqual([row["video_id"] for row in samples], ["pending"])
            self.assertEqual(samples[0]["title"], "Pending title")


if __name__ == "__main__":
    unittest.main()
