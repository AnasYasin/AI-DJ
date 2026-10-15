"""Does a search result carry the name of the record we asked for.

The run's name score (`legacy.track_fetcher._name_score`) can clear its acceptance line on the artist, the
version words and a "Topic" channel alone, with a different title: on 2026-09-24 "AADJA - Introvert Problems"
scored "aadja - psyhop" at 2.60, and "Syren (Amelie Lens Remix)" scored "Syren (Adam Sellouk Remix)" at 2.95.
Anas's rule: the track must have the same name and the right version, and the first result is not always
it. These three checks sit in front of the score when `download.strict_names` is on.
"""

import difflib
import re

from ..legacy.track_fetcher import _clean, _title_overlap

UNIDENTIFIED = re.compile(r"^\s*(id|unknown|untitled|\?+)(\s*[\(\[].*[\)\]])?\s*$", re.I)
GENERIC = {"original", "mix", "version", "extended", "edit", "remix", "rmx", "radio", "club", "dub", "vip",
           "instrumental", "remaster", "remastered", "feat", "ft", "featuring", "official", "video", "audio", "the"}
REMIX_WORDS = re.compile(r"\b(remix|rmx|bootleg|rework|edit|flip|refix)\b", re.I)
BRACKETS = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
TITLE_WORDS_MIN = 0.8      # share of the wanted title's words that must appear in the candidate's title
TITLE_RATIO_MIN = 0.85     # or the cleaned titles must be this similar end to end


def is_unidentified(title: str) -> bool:
    """'ID', 'Unknown', '???': there is nothing to search for."""
    return bool(UNIDENTIFIED.match(title or ""))


def remixer_words(title: str) -> set:
    """The names inside the brackets, minus the generic words: who made this version."""
    words = set()
    for inner in BRACKETS.findall(title):
        words |= {w for w in _clean(inner).split() if w not in GENERIC}
    return words


def title_matches(title: str, cand_title: str) -> bool:
    """The candidate carries the track's own title, brackets aside."""
    if _title_overlap(title, cand_title) >= TITLE_WORDS_MIN:
        return True
    return difflib.SequenceMatcher(None, _clean(title), _clean(cand_title)).ratio() >= TITLE_RATIO_MIN


def version_matches(title: str, cand_title: str) -> bool:
    """A remix is a different record from the original, and one remix from another."""
    want = remixer_words(title)
    if want:
        got = set(re.sub(r"[^a-z0-9]+", " ", cand_title.lower()).split())   # brackets kept: the remixer sits in them
        return len(want & got) / len(want) >= 0.8
    return not any(REMIX_WORDS.search(inner) for inner in BRACKETS.findall(cand_title))


def same_record(title: str, cand_title: str) -> bool:
    return title_matches(title, cand_title) and version_matches(title, cand_title)


def closeness(title: str, cand_title: str) -> float:
    """How close a candidate's title is to the one we asked for: title similarity, plus the remixer named in
    the brackets agreeing (or both having none). Used to pick the best of the top search results rather
    than to reject (Anas, 2026-09-24: "exact match is not a good way, the closest title wins")."""
    # the share of the wanted title's words present carries the decision, because a candidate's title also
    # holds the artist and a label tag ("Anyma & Rebuke - Syren [Drumcode]"), which a whole-string ratio
    # punishes; the ratio only breaks ties
    words = title_words_overlap(title, cand_title)
    ratio = difflib.SequenceMatcher(None, _clean(title), _clean(cand_title)).ratio()
    return words + (0.5 if version_matches(title, cand_title) else 0.0) + 0.2 * ratio


def title_words_overlap(title: str, cand_title: str) -> float:
    """Share of the wanted title's words (brackets aside, one-letter tokens dropped) in the candidate's
    title. "Don't Chat" is {dont... no: {don, chat} once 't' is dropped, so a title sharing only "don't"
    scores 0.5 and not 0.67."""
    want = {w for w in _clean(title).split() if len(w) > 1}
    got = {w for w in _clean(cand_title).split() if len(w) > 1}
    return len(want & got) / max(len(want), 1)


def search_queries(artist: str, title: str) -> list:
    """The query shapes worth sending. The full "artist title" first; then, when the title names a remixer
    or dub-maker in brackets, "title remixer" without the original artist, because that is how the upload
    is titled and SoundCloud's search finds nothing for the original artist's name (2026-09-24:
    "Elvis Crespo Suavemente (Oppidan Dub)" gave 0 results, "Oppidan Suavemente Dub" gave 5)."""
    queries = [f"{artist} {title}"]
    inner = " ".join(BRACKETS.findall(title)).strip()
    if inner:
        base = BRACKETS.sub(" ", title).strip()
        queries.append(f"{base} {inner}")
    return queries


ARTIST_SPLIT = re.compile(r"\s*(?:&|,|\+|/|\bx\b|\bvs\.?\b|\bfeat\.?\b|\bft\.?\b|\bpres\.?\b|\bw/|\band\b)\s*", re.I)
_WORDS = re.compile(r"[^a-z0-9]+")


def words(text: str) -> set:
    return set(_WORDS.sub(" ", (text or "").lower()).split())


def artists_of(credit: str) -> list:
    """'Oppidan & Hans Glader' -> [{'oppidan'}, {'hans', 'glader'}]; 'ft.' and 'feat.' names count too."""
    return [words(p) for p in ARTIST_SPLIT.split(credit or "") if words(p)]


def artist_present(credit: str, title: str, cand_title: str, channel: str) -> bool:
    """Is this upload by, or about, one of the people on the record (Anas, 2026-09-24).

    Any ONE listed artist fully in the candidate's title or channel is enough ("AH BEAT" on the channel
    Oppidan for "Oppidan & Hans Glader"). A channel that is one of an artist's names is enough. The
    remixer or dub-maker in the wanted title's brackets counts as an artist ("Suavemente (Oppidan Dub)"
    uploaded by Oppidan). Half of all the artist words present, the old rule, still passes."""
    seen = words(cand_title) | words(channel)
    chan = words(channel) - GENERIC - {"official", "records", "music", "tv", "topic"}
    people = artists_of(credit) + ([remixer_words(title)] if remixer_words(title) else [])
    for person in people:
        if person and person <= seen:
            return True
        if chan and chan <= person:
            return True
    every = set().union(*people) if people else set()
    return bool(every) and len(every & seen) / len(every) >= 0.5


def bad_title(title: str, cand_title: str, bad_pattern) -> bool:
    """The legacy bad-title words (radio edit, acapella, live set...) reject a candidate, unless the same
    word is in the title we asked for: when the DJ played the acappella, "acapella" is the record."""
    hits = {m.group(0).lower().replace(" ", "") for m in bad_pattern.finditer(cand_title or "")}
    wanted = {m.group(0).lower().replace(" ", "") for m in bad_pattern.finditer(title or "")}
    wanted |= {"acapella", "acappella"} if re.search(r"a\s*c+a?p+ella", (title or "").lower()) else set()
    hits = {h for h in hits if not (h.startswith("acap") and any(w.startswith("acap") for w in wanted))}
    return bool(hits - wanted)
