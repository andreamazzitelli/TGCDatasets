"""Shared helpers for the download_*.py scripts in this folder.

Every script drops images into data/<model>/train/<label>/ (relative to the
repo root) and appends one row per downloaded file to data/manifest.csv so
provenance stays tracked — see docs/data_collection.md.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
# `set` sits right after `tcg` (the set/card identifier is naturally "which
# tcg, which set" before you get to the free-text `source` string). Added
# as an additive column — see docs/tcg_detection_pipeline.md §2. Existing
# manifest rows predating this column were migrated once (blank `set`);
# `append_manifest` below always writes a value (possibly "") for it now.
# `detection_split` is appended last: scripts/build_detection_split.py owns
# assigning it for existing rows (by set, via a full-file rewrite), but a
# writer that already knows the right bucket for a new row (e.g.
# scripts/augment.py, which must keep an augmented copy in the same bucket
# as its source image) can pass it through append_manifest directly instead
# of leaving it blank for a later build_detection_split.py run to guess at.
MANIFEST_HEADER = ["filename", "model", "split", "label", "tcg", "set", "source", "license_note", "date_added", "notes", "detection_split"]

DEFAULT_HEADERS = {"User-Agent": "TGCDatasets-collector/1.0 (personal Create ML dataset build)"}

KNOWN_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic"}

_shape_warned: set[str] = set()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def ensure_manifest() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(MANIFEST_HEADER)


def append_manifest(*, filename: str, model: str, label: str, tcg: str, source: str, license_note: str, notes: str = "", split: str = "train", set: str = "", detection_split: str = "") -> None:
    ensure_manifest()
    with MANIFEST_PATH.open("a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([filename, model, split, label, tcg, set, source, license_note, date.today().isoformat(), notes, detection_split])


def suffix_from_url(url: str, default: str = ".jpg") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in KNOWN_IMAGE_SUFFIXES else default


def download_image(session: requests.Session, url: str, dest: Path, *, timeout: int = 20) -> bool:
    """Download `url` to `dest`. Returns True if a new file was written,
    False if it already existed or the download failed."""
    if dest.exists():
        return False
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! failed to download {url}: {exc}", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def sleep_polite(seconds: float) -> None:
    time.sleep(seconds)


def split_id_prefix(identifier: str, sep: str = "-") -> str:
    """Return the segment of `identifier` before the first `sep`, or "" if
    `sep` isn't present. Several sources format their per-card id as
    "<set>-<number>" (dbs, digimon, one_piece) or "<set>_<number>" (swu), so
    the set code is recoverable by splitting once from the left. Shared by
    those download scripts (called on the raw id at request time) and by
    scripts/backfill_manifest_sets.py (called on the id portion of the
    stored `source` column for pre-existing manifest rows, since the same
    ids end up there — see docs/tcg_detection_pipeline.md §2)."""
    return identifier.split(sep, 1)[0] if sep in identifier else ""


def first_present(d: dict, *keys: str):
    """Return the first non-empty value among the given keys, matched
    case-insensitively — handles APIs whose exact field casing we haven't
    verified live."""
    lower = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        v = lower.get(key.lower())
        if v:
            return v
    return None


def warn_once_unknown_shape(context: str, d: dict) -> None:
    """Print a one-time diagnostic when a script can't find the field it
    expected in an API response, so a run can be debugged from its output
    instead of failing silently."""
    if context in _shape_warned:
        return
    _shape_warned.add(context)
    print(f"  ! couldn't find an image field in a {context}; available keys: {sorted(d.keys())}", file=sys.stderr)
    print("    Report these keys back so the script's field names can be corrected.", file=sys.stderr)
