import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

from audio_extractor import middle_sample_times
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


if __name__ == "__main__":
    unittest.main()
