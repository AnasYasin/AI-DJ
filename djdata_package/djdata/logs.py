"""Logging for every stage: console plus one file under root/logs, one line per event with the
worker name (thread or process), the stage, the item id and the elapsed time."""

import logging
from pathlib import Path
import time

FORMAT = "%(asctime)s %(levelname)-7s %(threadName)s/%(processName)s %(name)s: %(message)s"


def setup(log_dir: Path, level: str = "INFO", name: str = "djdata") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)
        fmt = logging.Formatter(FORMAT)
        con = logging.StreamHandler()
        con.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(con)
        root.addHandler(fh)
    for noisy in ("yt_dlp", "urllib3", "numba", "nodriver", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


class Timed:
    """`with Timed(log, "stage", item) as t:` logs start and end with elapsed seconds and outcome."""

    def __init__(self, log: logging.Logger, stage: str, item: str):
        self.log, self.stage, self.item = log, stage, item

    def __enter__(self):
        self.t0 = time.time()
        self.log.info("%s start %s", self.stage, self.item)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc is None:
            self.log.info("%s done %s in %.1fs", self.stage, self.item, dt)
        else:
            self.log.error("%s FAILED %s after %.1fs: %s", self.stage, self.item, dt, exc)
        return False
