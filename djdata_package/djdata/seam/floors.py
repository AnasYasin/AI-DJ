"""One rule for every vote floor in the seam measurement: measure the noise, do not assume it.

Why this exists. Every threshold in the pipeline was a constant somebody picked. `MIN_VOTES = 40` in
locate, `VERIFY_MIN_HASHES = 50` in the fetcher, `VOTE_FLOOR = 6` in the band and bass scripts. Those
numbers came from one observation, that a wrong record scored 21 on the fetcher and 14 on a seam window.
They hold on the material that observation was made on and nowhere else.

Measured 2026-09-21 on Fred again's marathon, which YouTube served at 48 kbps HE-AAC: three records by a
completely different DJ, fed in as controls, scored 43, 47 and 58 votes. The floor in the code was 40, so
it sat underneath the noise and every reading between 30 and 80 was accepted as a match. On Raveform's
audio the same control sits at 13. No single constant can serve both.

The rule. A floor is a property of one seam in one band on one piece of audio, so it is measured there:
several records that are definitely not in this mix go through the identical path, and the floor is a
margin above the loudest of them. Several, not one, because one control is a single draw and the spread
across Fred's three was 15 votes.

Every caller records `control_n`, `control_max` and `floor` next to its result, so a row carries the
evidence for its own threshold and a bad floor is visible rather than silent.
"""

from dataclasses import asdict, dataclass
import math

MIN_CONTROLS = 3  # one control is a single draw of the noise, not a distribution
MARGIN = 2.0  # a real match must beat the loudest wrong record by this factor
ABSOLUTE_MIN = 4  # below a handful of votes a count is noise whatever the control did


@dataclass(frozen=True)
class Floor:
    """A threshold and the evidence for it."""

    floor: int
    control_max: int
    control_n: int
    control_votes: tuple
    margin: float
    trusted: bool
    reason: str = ""

    def clears(self, votes) -> bool:
        return votes >= self.floor

    def as_columns(self, prefix: str = "") -> dict:
        row = asdict(self)
        row["control_votes"] = " ".join(str(v) for v in self.control_votes)
        return {f"{prefix}{k}": v for k, v in row.items()}


def from_controls(
    control_votes,
    margin: float = MARGIN,
    absolute_min: int = ABSOLUTE_MIN,
    min_controls: int = MIN_CONTROLS,
) -> Floor:
    """The floor for one seam, one band, from records that are not in this mix.

    `control_votes` is the best vote count each control record scored anywhere in the window, by the
    same code path the real records take. Fewer than `min_controls` still returns a usable floor, marked
    untrusted, so a caller can carry on and the row says the threshold was thin.
    """
    votes = tuple(int(v) for v in control_votes)
    if not votes:
        return Floor(
            floor=absolute_min,
            control_max=0,
            control_n=0,
            control_votes=(),
            margin=margin,
            trusted=False,
            reason="no control ran",
        )
    highest = max(votes)
    floor = max(absolute_min, int(math.ceil(margin * highest)))
    trusted = len(votes) >= min_controls
    reason = "" if trusted else f"only {len(votes)} controls, {min_controls} wanted"
    return Floor(
        floor=floor,
        control_max=highest,
        control_n=len(votes),
        control_votes=votes,
        margin=margin,
        trusted=trusted,
        reason=reason,
    )


def separation(real_votes, floor: Floor) -> float:
    """How far the real record sits above its own floor. Under 1 means it did not clear it.

    This is the number that says whether a measurement is safe or lucky. A seam where the right record
    scores 900 against a floor of 28 is settled. One where it scores 30 against 28 is a coin toss, and
    the two should never be reported as the same thing.
    """
    best = max(real_votes) if hasattr(real_votes, "__iter__") else real_votes
    return round(float(best) / max(floor.floor, 1), 2)
