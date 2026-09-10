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
MANIFEST_HEADER = ["filename", "model", "label", "tcg", "source", "license_note", "date_added", "notes"]

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
            csv.writer(f).writerow(MANIFEST_HEADER)


def append_manifest(*, filename: str, model: str, label: str, tcg: str, source: str, license_note: str, notes: str = "") -> None:
    ensure_manifest()
    with MANIFEST_PATH.open("a", newline="") as f:
        csv.writer(f).writerow([filename, model, label, tcg, source, license_note, date.today().isoformat(), notes])


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
