"""Read YouTube URLs from an existing dataset without modifying it."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


URL_COLUMN_CANDIDATES = (
    "video_url",
    "youtube_url",
    "webpage_url",
    "url",
    "link",
)


def _looks_like_youtube_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        host = urlparse(value.strip()).netloc.lower()
    except ValueError:
        return False
    return host == "youtu.be" or host.endswith("youtube.com")


def _looks_like_youtube_video_url(value: Any) -> bool:
    if not _looks_like_youtube_url(value):
        return False
    parsed = urlparse(str(value).strip())
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host == "youtu.be":
        return bool(path.strip("/"))
    return path == "/watch" or path.startswith(("/shorts/", "/live/", "/embed/"))


def identify_url_column(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    """Prefer known names, then inspect values instead of assuming `url`."""
    lowered = {name.lower().strip(): name for name in fieldnames}
    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    for name in fieldnames:
        if any(_looks_like_youtube_url(row.get(name)) for row in rows[:100]):
            return name
    raise ValueError("No column containing YouTube URLs could be identified.")


def _sqlite_connection_read_only(path: Path) -> sqlite3.Connection:
    # mode=ro guarantees that the benchmark cannot write to the source database.
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect_dataset(dataset_path: str | Path) -> dict[str, Any]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        with closing(_sqlite_connection_read_only(path)) as connection:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            matches: list[dict[str, Any]] = []
            for table in tables:
                safe_table = table.replace('"', '""')
                columns = [
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{safe_table}")')
                ]
                sample_rows = [
                    dict(zip(columns, row))
                    for row in connection.execute(
                        f'SELECT * FROM "{safe_table}" LIMIT 100'
                    ).fetchall()
                ]
                try:
                    url_column = identify_url_column(columns, sample_rows)
                except ValueError:
                    continue
                row_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{safe_table}"'
                ).fetchone()[0]
                matches.append({
                    "path": str(path),
                    "format": "sqlite",
                    "table": table,
                    "url_column": url_column,
                    "columns": columns,
                    "row_count": row_count,
                    "filter": "is_darija = 1" if "is_darija" in columns else None,
                })
            if matches:
                # A database can contain both channel and video URLs. Rank all
                # matches so `video_url` wins regardless of table name order.
                def rank(match: dict[str, Any]) -> tuple[int, int]:
                    column = str(match["url_column"]).lower()
                    column_rank = URL_COLUMN_CANDIDATES.index(column) if column in URL_COLUMN_CANDIDATES else 99
                    table_rank = 0 if str(match["table"]).lower() in {"videos", "video"} else 1
                    return column_rank, table_rank
                return min(matches, key=rank)
        raise ValueError("No SQLite table containing YouTube URLs was found.")

    rows = list(_read_simple_rows(path, limit=100))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    columns = list(rows[0].keys())
    return {
        "path": str(path),
        "format": suffix.lstrip("."),
        "table": None,
        "url_column": identify_url_column(columns, rows),
        "columns": columns,
        "row_count": None,
        "filter": None,
    }


def _read_simple_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from _limited(csv.DictReader(handle), limit)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for key in ("videos", "items", "data", "rows"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list, or a videos/items/data/rows list.")
        yield from _limited((row for row in data if isinstance(row, dict)), limit)
    elif suffix in {".jsonl", ".ndjson"}:
        def parsed_lines() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            yield row
        yield from _limited(parsed_lines(), limit)
    elif suffix == ".txt":
        def text_rows() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        yield {"video_url": line.strip()}
        yield from _limited(text_rows(), limit)
    else:
        raise ValueError("Supported datasets: SQLite, CSV, JSON, JSONL/NDJSON, TXT.")


def _limited(rows: Iterator[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            return
        yield row


def iter_youtube_records(dataset_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield unique records in source order; the source remains read-only."""
    path = Path(dataset_path).expanduser().resolve()
    info = inspect_dataset(path)
    url_column = info["url_column"]
    seen: set[str] = set()

    if info["format"] == "sqlite":
        table = str(info["table"]).replace('"', '""')
        columns = info["columns"]
        where_clause = ' WHERE "is_darija" = 1' if "is_darija" in columns else ""
        with closing(_sqlite_connection_read_only(path)) as connection:
            cursor = connection.execute(f'SELECT * FROM "{table}"{where_clause}')
            for values in cursor:
                row = dict(zip(columns, values))
                url = str(row.get(url_column) or "").strip()
                if _looks_like_youtube_video_url(url) and url not in seen:
                    seen.add(url)
                    yield {"youtube_url": url, "source_row": row}
        return

    for row in _read_simple_rows(path):
        url = str(row.get(url_column) or "").strip()
        if _looks_like_youtube_video_url(url) and url not in seen:
            seen.add(url)
            yield {"youtube_url": url, "source_row": row}
