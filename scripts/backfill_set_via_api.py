#!/usr/bin/env python3
"""Metadata-only backfill of the `set` column for pre-existing
magic_the_gathering and yugioh manifest rows whose `set` is still blank.

scripts/backfill_manifest_sets.py already established that these two tcgs
are NOT derivable from the stored `source` string alone (Scryfall's source
is an opaque card UUID; YGOPRODeck's source is its own numeric card id, both
with no visible set code) — see that script's module docstring. This script
closes that gap differently: instead of re-deriving from what's already
stored, it looks the real set up against the same provider APIs
download_magic.py / download_yugioh.py already use, keyed by the id already
recorded in `source`.

This does NOT re-download any image — it only patches the `set` column of
existing manifest rows. For both providers this uses a single bulk metadata
fetch (mirroring how download_magic.py and download_yugioh.py themselves
fetch once, not per-card) rather than one HTTP request per card, which is
both far more polite and far faster than ~300+300 individual lookups:

- magic_the_gathering: Scryfall's own bulk `default_cards` file (same
  endpoint download_magic.py uses) already carries each card's `set` code,
  so it's fetched once and used as a local id -> set lookup table. Cached
  under $ML_CACHE_DIR (see below) so a re-run (e.g. after this script
  partially fails) doesn't refetch the ~100+MB bulk file twice.
- yugioh: YGOPRODeck's bulk `cardinfo.php` (same endpoint download_yugioh.py
  uses, no id filter) returns every card in one response; each card's
  `card_sets[0].set_code` is used, mirroring download_yugioh.py's own
  set-derivation logic exactly. Also cached.

Per CLAUDE.md / the user's instruction: any new bulk cache file goes on the
external volume this repo's data/ already symlinks to, never under this
repo or /tmp — see ML_CACHE_DIR below.

Usage:
    python3 scripts/backfill_set_via_api.py --dry-run   # report only
    python3 scripts/backfill_set_via_api.py              # write for real

Writing for real first copies data/manifest.csv to data/manifest.csv.bak
(overwriting any previous backup), same convention as
scripts/backfill_manifest_sets.py.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "download"))
from common import make_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
BACKUP_PATH = REPO_ROOT / "data" / "manifest.csv.bak"

# Same external volume this repo's data/ symlinks already resolve to (see
# data/tcg_identifier -> /Volumes/Extreme SSD/TCGCollector/ml_datasets/...).
# Bulk provider dumps are sizeable (Scryfall's default_cards is 100+MB) and
# not bulk-redistribution data this repo tracks, so they're cached here
# rather than under this repo or /tmp.
ML_CACHE_DIR = Path("/Volumes/Extreme SSD/TCGCollector/ml_datasets/_cache")
SCRYFALL_CACHE = ML_CACHE_DIR / "scryfall_default_cards.json.gz"
YGOPRODECK_CACHE = ML_CACHE_DIR / "ygoprodeck_cardinfo.json.gz"

SCRYFALL_BULK_INDEX_URL = "https://api.scryfall.com/bulk-data/default_cards"
YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"


def source_id(source: str) -> str:
    return source.split(":", 1)[1] if ":" in source else ""


def _load_or_fetch(cache_path: Path, fetch) -> object:
    """Return cached JSON if present, else call `fetch()`, cache it, and
    return that. Caching means a second run (e.g. after a partial failure)
    doesn't hit the provider's bulk endpoint twice."""
    if cache_path.exists():
        print(f"  using cached {cache_path}")
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = fetch()
    ML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"  cached to {cache_path}")
    return data


def build_scryfall_set_map(session) -> dict[str, str]:
    def fetch():
        print(f"Fetching Scryfall bulk-data index from {SCRYFALL_BULK_INDEX_URL} ...")
        index = session.get(SCRYFALL_BULK_INDEX_URL, timeout=30).json()
        download_uri = index.get("download_uri") or index.get("jsonl_download_uri")
        if not download_uri:
            sys.exit(f"Unexpected response from Scryfall, no download_uri field: {index}")
        print(f"Downloading card list from {download_uri} (large file, may take a minute) ...")
        raw = session.get(download_uri, timeout=180).content
        if download_uri.endswith(".gz"):
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8")
        if download_uri.endswith(".jsonl") or download_uri.endswith(".jsonl.gz"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)

    cards = _load_or_fetch(SCRYFALL_CACHE, fetch)
    print(f"  {len(cards)} Scryfall card entries loaded.")
    return {c["id"]: (c.get("set", "") or "") for c in cards if c.get("id")}


def build_ygoprodeck_set_map(session) -> dict[str, str]:
    def fetch():
        print(f"Fetching YGOPRODeck full card list from {YGOPRODECK_URL} ...")
        resp = session.get(YGOPRODECK_URL, timeout=60).json()
        return resp.get("data", [])

    cards = _load_or_fetch(YGOPRODECK_CACHE, fetch)
    print(f"  {len(cards)} YGOPRODeck card entries loaded.")
    id_to_set: dict[str, str] = {}
    for c in cards:
        cid = c.get("id")
        if cid is None:
            continue
        # Mirror download_yugioh.py exactly: first card_sets[0].set_code,
        # not a "correct" printing, just the same deterministic pick the
        # live download script already uses.
        card_sets = c.get("card_sets") or []
        set_code = card_sets[0].get("set_code", "") if card_sets else ""
        id_to_set[str(cid)] = set_code
    return id_to_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        parser.error(f"No manifest found at {MANIFEST_PATH}")

    with MANIFEST_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "set" not in fieldnames:
        parser.error("manifest has no `set` column yet")

    targets = {"magic_the_gathering", "yugioh"}
    blank_rows = [r for r in rows if (r.get("tcg") or "") in targets and not (r.get("set") or "").strip()]
    if not blank_rows:
        print("No magic_the_gathering/yugioh rows with a blank `set` — nothing to do.")
        return

    session = make_session()

    need_scryfall = any(r["tcg"] == "magic_the_gathering" for r in blank_rows)
    need_ygoprodeck = any(r["tcg"] == "yugioh" for r in blank_rows)

    scryfall_map: dict[str, str] = build_scryfall_set_map(session) if need_scryfall else {}
    ygoprodeck_map: dict[str, str] = build_ygoprodeck_set_map(session) if need_ygoprodeck else {}

    backfilled = Counter()
    not_found = Counter()
    empty_in_api = Counter()  # id found, but API's own set field was empty
    not_found_ids: dict[str, list[str]] = {"magic_the_gathering": [], "yugioh": []}
    changed = 0

    for row in blank_rows:
        tcg = row["tcg"]
        cid = source_id(row.get("source", "") or "")
        lookup = scryfall_map if tcg == "magic_the_gathering" else ygoprodeck_map
        if cid not in lookup:
            not_found[tcg] += 1
            not_found_ids[tcg].append(cid)
            continue
        set_value = lookup[cid]
        if not set_value:
            empty_in_api[tcg] += 1
            continue
        row["set"] = set_value
        backfilled[tcg] += 1
        changed += 1

    total_blank = len(blank_rows)
    print(f"\n{total_blank} magic_the_gathering/yugioh row(s) had a blank `set` before this backfill.\n")
    print(f"{'tcg':<22}{'blank_before':<14}{'backfilled':<12}{'not_found':<12}{'found_but_empty'}")
    print("-" * 80)
    for tcg in sorted(targets):
        blank_n = sum(1 for r in blank_rows if r["tcg"] == tcg)
        if blank_n == 0:
            continue
        print(f"{tcg:<22}{blank_n:<14}{backfilled.get(tcg, 0):<12}{not_found.get(tcg, 0):<12}{empty_in_api.get(tcg, 0)}")

    for tcg in sorted(targets):
        if not_found_ids[tcg]:
            sample = not_found_ids[tcg][:5]
            print(f"\n  {tcg}: {len(not_found_ids[tcg])} id(s) not found in the current API response "
                  f"(likely removed/renamed since the original download) — sample: {sample}")

    print(f"\n{changed} row(s) backfilled. {total_blank - changed} row(s) remain blank.")

    if args.dry_run:
        print("\nDry run — manifest not written.")
        return

    if changed == 0:
        print("\nNothing changed — manifest not rewritten.")
        return

    shutil.copy2(MANIFEST_PATH, BACKUP_PATH)
    print(f"\nBacked up {MANIFEST_PATH} -> {BACKUP_PATH}")

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
