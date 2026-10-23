"""Buffered JSONL output for inference instrumentation."""

import atexit
import json
from pathlib import Path


class _JsonlWriter:
    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self.rows_since_flush = 0

    def write(self, record: dict) -> None:
        self.handle.write(
            json.dumps(record, separators=(",", ":"), default=str) + "\n"
        )
        self.rows_since_flush += 1
        if self.rows_since_flush >= 128:
            self.handle.flush()
            self.rows_since_flush = 0

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()


_writers = {}


def write_jsonl_record(path, record: dict) -> None:
    """Append one compact JSON row without reopening the file each batch."""
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(trace_path.resolve())
    writer = _writers.get(key)
    if writer is None or writer.handle.closed:
        writer = _JsonlWriter(trace_path)
        _writers[key] = writer
    writer.write(record)


def close_jsonl_writers() -> None:
    """Flush open trace files, primarily for process shutdown and tests."""
    for writer in _writers.values():
        writer.close()
    _writers.clear()


atexit.register(close_jsonl_writers)
