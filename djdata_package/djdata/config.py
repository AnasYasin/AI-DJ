"""Load config.yaml into a plain object. One place knows the file layout under `root`."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    raw: dict
    root: Path
    source: str
    raveform_dir: Path
    tracklists_csv: Path
    run_tiers: list
    tiers: dict
    usable: dict
    workers: dict
    download: dict
    window: dict
    analysis: dict
    log_level: str = "INFO"
    dirs: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.root / "state.sqlite"


def load(path: str | Path) -> Config:
    raw = yaml.safe_load(open(path))
    root = Path(raw["root"])
    cfg = Config(
        raw=raw,
        root=root,
        source=raw["source"],
        raveform_dir=Path(raw["raveform_dir"]),
        tracklists_csv=Path(raw.get("tracklists_csv", "")),
        run_tiers=list(raw["run_tiers"]),
        tiers=raw["tiers"],
        usable=raw["usable"],
        workers=raw["workers"],
        download=raw["download"],
        window=raw["window"],
        analysis=raw["analysis"],
        log_level=raw.get("logging", {}).get("level", "INFO"),
    )
    cfg.dirs = {
        "mixes_tmp": root / "mixes_tmp",   # full mixes, deleted once their windows are cut
        "windows": root / "windows",       # one audio file per seam
        "tracks": root / "tracks",         # one audio file per track id
        "out": root / "out",               # seams.jsonl, curves/, exports
        "curves": root / "out" / "curves",
        "logs": root / "logs",
    }
    for d in cfg.dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return cfg
