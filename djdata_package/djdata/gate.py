"""Pacing and block handling shared by the download workers.

pace(): at most one download start every `min_interval_s` across all workers. The yt-dlp wiki puts a
guest session at about 300 videos per hour; 20 s spacing is 180 per hour, with margin.

blocked(): the worker that hit a YouTube block calls this. Every worker then waits in wait_open().
A recovery thread probes YouTube every `block_wait_s` and reopens the gate when a probe succeeds.
Mix downloads from other hosts and seam analysis are not held.
"""

import logging
import threading
import time

log = logging.getLogger("djdata.gate")


class Gate:
    def __init__(self, min_interval_s: float, block_wait_s: float, probe, stop: threading.Event,
                 clock=time.monotonic, sleep=time.sleep):
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
        threading.Thread(target=self._recover, name="gate-recover", daemon=True).start()

    def _recover(self):
        while not self._stop.is_set():
            if self._stop.wait(self.block_wait_s):
                return
            if self._probe():
                log.warning("YouTube block lifted; downloads resume")
                self._open.set()
                return
            log.warning("YouTube still blocked; next probe in %.0f s", self.block_wait_s)
