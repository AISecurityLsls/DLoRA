from __future__ import annotations

import os
import sys
import traceback


class Tee:
    """Mirror stdout/stderr to both terminal and a log file."""
    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        log_dir = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(log_dir, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = _StreamTee(self.stdout, self.file)
        sys.stderr = _StreamTee(self.stderr, self.file)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and self.file is not None:
                self.file.write("\n[UNCAUGHT EXCEPTION]\n")
                traceback.print_exception(exc_type, exc, tb, file=self.file)
                self.file.flush()
        finally:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            if self.file is not None:
                self.file.close()


class _StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()
