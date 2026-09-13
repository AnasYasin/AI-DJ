"""Pacing and block handling shared by the download workers.

pace(): at most one download start every `min_interval_s` across all workers. The yt-dlp wiki puts a
guest session at about 300 videos per hour; 20 s spacing is 180 per hour, with margin.

blocked(): the worker that hit a YouTube block calls this. Every worker then waits in wait_open().
A recovery thread probes YouTube every `block_wait_s` and reopens the gate when a probe succeeds.
Mix downloads from other hosts and seam analysis are not held.

State file: every change is written to `status_path` as JSON (state, since, blocks, last_probe), so
`djdata status` and a plain `cat` over ssh show whether downloads are running or paused.
"""

import json
import logging
from pathlib import Path
import threading
import time

log = logging.getLogger("djdata.gate")


class Gate:
    def __init__(self, min_interval_s: float, block_wait_s: float, probe, stop: threading.Event,
                 clock=time.monotonic, sleep=time.sleep, status_path: Path | None = None):
        self.status_path = Path(status_path) if status_path else None
        self.min_interval_s = min_interval_s
        self.block_wait_s = block_wait_s
        self._probe = probe
        self._stop = stop
        self._clock = clock
        self._sleep = sleep
        self._pace_lock = threading.Lock()
        self._next_start = 0.0
        self._open = threading.Event()
        self._open.set()
        self.blocks = 0
        self._write("open", reason="run start")

    def _write(self, state: str, **extra):
        if self.status_path is None:
            return
        now = time.time()
        row = {"state": state, "since": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)), "since_epoch": now,
               "blocks": self.blocks, **extra}
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1))
        tmp.replace(self.status_path)

    def pace(self):
        """Block the caller until its download may start, then reserve the next slot."""
        with self._pace_lock:
            wait = self._next_start - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_start = self._clock() + self.min_interval_s

    def is_open(self) -> bool:
        return self._open.is_set()

    def wait_open(self):
        while not self._open.is_set() and not self._stop.is_set():
            self._open.wait(1.0)

    def blocked(self, reason: str):
        """Close the gate and start probing. A second caller during the same block is a no-op."""
        if not self._open.is_set():
            return
        self._open.clear()
        self.blocks += 1
        log.warning("YouTube block #%d (%s); downloads paused, probing every %.0f s", self.blocks, reason, self.block_wait_s)
        self._write("blocked", reason=reason[:200])
        threading.Thread(target=self._recover, name="gate-recover", daemon=True).start()

    def _recover(self):
        while not self._stop.is_set():
            if self._stop.wait(self.block_wait_s):
                return
            if self._probe():
                log.warning("YouTube block lifted; downloads resume")
                self._write("open", reason="probe passed")   # file first, so a reader never sees open workers with a 'blocked' file
                self._open.set()
                return
            log.warning("YouTube still blocked; next probe in %.0f s", self.block_wait_s)
            self._write("blocked", reason="probe failed", last_probe=time.strftime("%H:%M:%S UTC", time.gmtime()))
