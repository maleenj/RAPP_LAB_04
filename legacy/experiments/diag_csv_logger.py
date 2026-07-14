"""Lightweight CSV logger for diagnostic nodes.

Creates timestamped CSV files with automatic header writing and periodic
flushing.  Designed for 15-30 Hz control loops — flushes every N rows
(default 10) so data survives crashes without per-row I/O overhead.
"""

import csv
import os
import time
from pathlib import Path


class DiagCSVLogger:
    """Write diagnostic rows to a timestamped CSV file."""

    def __init__(
        self,
        prefix: str,
        columns: list[str],
        output_dir: str = "/data/processed/diagnostics",
        label: str = "",
        flush_interval: int = 10,
    ):
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        filename = f"{prefix}{suffix}_{ts}.csv"
        self._path = Path(output_dir) / filename

        self._columns = columns
        self._flush_interval = flush_interval
        self._row_count = 0

        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(columns)
        self._file.flush()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, row: list):
        """Append a row (values must match column order)."""
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % self._flush_interval == 0:
            self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()
