"""The dataset tables. Flat CSV files under a corpus's out/ folder, one per stage, append only.

`state.sqlite` is the pipeline's resume log and nobody reads it. These tables are what the notebooks,
the rule making and the model training read, so they stay plain: one header, one row per key, every
number next to the control floor that judged it, times in seconds.

A stage resumes by asking its table which keys are already there and skipping those. A stage that must
redo a row deletes it first with `drop`. Nothing is updated in place.
"""

import csv
import logging
from pathlib import Path

log = logging.getLogger("djdata.store.tables")


class Table:
    def __init__(self, path: Path, columns: list[str], key: str | tuple[str, ...]):
        self.path = Path(path)
        self.columns = list(columns)
        self.key = (key,) if isinstance(key, str) else tuple(key)

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def rows(self) -> list[dict]:
        if not self.exists():
            return []
        with open(self.path, newline="") as handle:
            return list(csv.DictReader(handle))

    def keys(self) -> set:
        return {self._key_of(r) for r in self.rows()}

    def _key_of(self, row: dict):
        vals = tuple(str(row[k]) for k in self.key)
        return vals[0] if len(vals) == 1 else vals

    def append(self, rows: list[dict]) -> int:
        """Write rows with exactly the table's columns. Missing values become empty, extras are an error
        so a stage cannot silently widen the table."""
        if not rows:
            return 0
        extra = set().union(*(r.keys() for r in rows)) - set(self.columns)
        if extra:
            raise ValueError(f"{self.path.name}: columns not in the table: {sorted(extra)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.exists()
        with open(self.path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            if new:
                writer.writeheader()
            for r in rows:
                writer.writerow({c: _cell(r.get(c)) for c in self.columns})
            handle.flush()
        log.debug("%s: appended %d rows", self.path.name, len(rows))
        return len(rows)

    def drop(self, keys: set) -> int:
        """Rewrite the file without the rows whose key is in `keys`. Returns how many went."""
        rows = self.rows()
        keep = [r for r in rows if self._key_of(r) not in keys]
        if len(keep) == len(rows):
            return 0
        with open(self.path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(keep)
        return len(rows) - len(keep)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value == value else ""
    return value


def num(value, default=None):
    """A CSV cell back to a number, or `default` when it is empty."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except ValueError:
        return default
    return int(f) if f.is_integer() and "." not in str(value) else f


PLAYS = [
    "mix_id",
    "dj",
    "genre",
    "mix_title",
    "mix_minutes",
    "track_id",
    "title",
    "is_control",
    "order_listed",
    "listed_min",
    "play_type",
    "overlay_parent",
    "time_zero_s",
    "rate",
    "votes",
    "floor",
    "control_max",
    "control_n",
    "sections",
    "sections_agree",
    "confidence",
    "found",
    "drift_min",
    "first_heard_s",
    "last_heard_s",
    "played_from_s",
    "played_to_s",
    "sweep_votes",
    "sweep_floor",
    "track_len_s",
    "seconds",
]


def plays(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "plays.csv", PLAYS, key=("mix_id", "track_id"))


SEAMS = [
    "seam_id",
    "mix_id",
    "dj",
    "genre",
    "track_a",
    "track_b",
    "order_a",
    "order_b",
    "order_ok",
    "listed_between",
    "unlocated_between",
    "gap_s",
    "usable",
    "reason",
    "a_confidence",
    "b_confidence",
    "a_time_zero_s",
    "a_rate",
    "a_first_heard_s",
    "a_last_heard_s",
    "a_end_s",
    "b_time_zero_s",
    "b_rate",
    "b_first_heard_s",
    "b_last_heard_s",
    "third_records",
    "window_t0",
    "window_t1",
    "window_s",
]

CUTS = [
    "seam_id",
    "window_file",
    "asked_s",
    "actual_s",
    "cut_error_s",
    "status",
    "audit_start_a",
    "audit_start_b",
    "audit_end_a",
    "audit_end_b",
    "audit_floor",
    "audit_control_max",
    "audit_start_ok",
    "audit_end_ok",
]

MEASURES = ["seam_id", "window_s", "control_n", "track_slice_s"]
for _band in ("low", "mid", "high"):
    MEASURES += [
        f"{_band}_out_s",
        f"{_band}_in_s",
        f"{_band}_a_votes",
        f"{_band}_b_votes",
        f"{_band}_floor",
        f"{_band}_control_max",
        f"{_band}_a_separation",
        f"{_band}_b_separation",
        f"{_band}_floor_trusted",
    ]
MEASURES += [
    "bass_swap_s",
    "bass_a_votes",
    "bass_b_votes",
    "bass_floor",
    "bass_control_max",
    "bass_a_separation",
    "bass_b_separation",
    "bass_floor_trusted",
    "in_s",
    "out_s",
    "overlap_s",
    "presence_floor",
    "presence_a_votes",
    "presence_b_votes",
    "loop_steps",
    "loop_floor",
    "bass_outgoing_s",
    "bass_both_s",
    "bass_incoming_s",
    "bass_cut_s",
    "bass_extra_s",
    "bass_break_s",
    "bass_unsure_s",
    "bass_cut_longest_s",
    "bass_both_longest_s",
    "a_best_separation",
    "b_best_separation",
    "measured",
    "control_trusted",
    "seconds",
]

TEMPOS = ["track_id", "bpm", "seconds"]

LABELS = [
    "seam_id",
    "label",
    "labels",
    "bar_s",
    "overlap_bars",
    "bass_cut_bars",
    "bass_both_bars",
    "sweep_out_bars",
    "sweep_in_bars",
    "loop_steps",
]


def seams(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "seams.csv", SEAMS, key="seam_id")


def cuts(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "cuts.csv", CUTS, key="seam_id")


def measures(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "measures.csv", MEASURES, key="seam_id")


def tempos(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "tempos.csv", TEMPOS, key="track_id")


def labels(out_dir: Path) -> Table:
    return Table(Path(out_dir) / "labels.csv", LABELS, key="seam_id")
