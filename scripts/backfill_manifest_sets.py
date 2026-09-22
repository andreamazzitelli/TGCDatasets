#!/usr/bin/env python3
"""Backfill the `set` column for pre-existing data/manifest.csv rows.

`set` was added to the manifest (see scripts/download/common.py and
docs/tcg_detection_pipeline.md §2) and threaded through each download_*.py
script *going forward*, but the ~3600 rows downloaded before that change all
have a blank `set`. That defeats scripts/build_detection_split.py's by-set
splitting for those rows: with `set` blank, build_detection_split.py treats
every such row as its own singleton "set" for split purposes, so images from
the same real-world set can land in both train and eval — letting a
classifier partly learn set-specific visual cues (border art, set symbol,
holofoil pattern) instead of true TCG-identifying cues, and inflating eval
accuracy relative to genuine unseen-set generalization.

This script does NOT re-fetch anything from any provider. It only re-derives
`set` from what's already stored in each row's `source` column, using
per-source logic equivalent to what each download_*.py script now applies at
request time. Some sources are fully derivable this way — their id format
was always "<set>-<number>" or "<set>_<number>" (dbs, digimon, one_piece,
pokemon, swu), so the same split that download_*.py applies to the raw id
also works applied to the id portion of `source`. One source (weiss_schwarz)
is only *approximately* derivable: the live download script's `set` comes
from the JSON filename it fetched, which isn't recoverable from `source`
alone, but the card's own `code` field (which *is* the id in `source`) is
formatted "<code-ish prefix>-<number>" and empirically has exactly one
hyphen across the whole manifest, so splitting it groups cards by that
prefix — not byte-identical to the live script's grouping, but a real,
consistent set-like grouping for backfill purposes, which is what
build_detection_split.py needs (see README note in the summary this script
prints). Some sources are genuinely NOT derivable from `source` alone
because they store only an opaque id with no visible set (magic_the_gathering
— Scryfall UUID; yugioh — YGOPRODeck's own numeric card id, unrelated to the
arbitrary card_sets[0] set code chosen at download time; flesh_and_blood —
opaque alphanumeric unique_id; lorcana — opaque "crd_<hex>" id). Those rows
are left blank, not guessed. `other_tcg` rows (grand_archive,
marvel_champions, netrunner), `not_a_card`, and any ebay_raw/ebay_graded/
psa_certs rows are left blank too, as already established elsewhere as
genuinely unavailable.

Usage:
    python3 scripts/backfill_manifest_sets.py --dry-run   # report only
    python3 scripts/backfill_manifest_sets.py              # write for real

Writing for real first copies data/manifest.csv to data/manifest.csv.bak
(overwriting any previous backup) so the backfill is reversible.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "download"))
from common import split_id_prefix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
BACKUP_PATH = REPO_ROOT / "data" / "manifest.csv.bak"

# tcg -> separator used to split the id portion of `source` into
# (set-ish prefix, remainder). Mirrors each download_*.py script's own
# parsing of the same id format at request time (see module docstring for
# the one approximation, weiss_schwarz).
DERIVABLE_TCGS: dict[str, str] = {
    "dragon_ball_super": "-",
    "digimon": "-",
    "one_piece": "-",
    "pokemon": "-",
    "star_wars_unlimited": "_",
    "weiss_schwarz": "-",  # approximate — see module docstring
}

# tcg values confirmed NOT derivable from the stored `source` string alone
# (checked against real manifest rows, not assumed) — reported, not guessed.
NOT_DERIVABLE_REASONS: dict[str, str] = {
    "magic_the_gathering": "source is an opaque Scryfall card UUID with no visible set code",
    "yugioh": "source is YGOPRODeck's own numeric card id; the set code used at download time came "
    "from an arbitrary card_sets[0] pick that isn't reflected in the id",
    "flesh_and_blood": "source is an opaque alphanumeric unique_id with no visible set code",
    "lorcana": "source is an opaque Lorcast \"crd_<hex>\" id; the set code used at download time came "
    "from a separate per-set API fetch not reflected in the id",
}


def source_id(source: str) -> str:
    """The id portion of a `source` value ("domain:id" -> "id")."""
    return source.split(":", 1)[1] if ":" in source else ""


def derive_set(tcg: str, source: str) -> str | None:
    """Return a derived `set` value for one row, or None if this tcg's
    source format isn't derivable at all (as opposed to derivable-but-this-
    particular-row-didn't-match, which returns "")."""
    sep = DERIVABLE_TCGS.get(tcg)
    if sep is None:
        return None
    return split_id_prefix(source_id(source), sep)


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

    blank_before = Counter()
    backfilled = Counter()
    still_blank_derivable = Counter()  # derivable tcg, but this row's id didn't match the expected shape
    still_blank_not_derivable = Counter()
    changed = 0

    for row in rows:
        existing = (row.get("set") or "").strip()
        if existing:
            continue  # already has a set (either backfilled already, or written by an updated download_*.py)
        tcg = row.get("tcg", "") or ""
        blank_before[tcg] += 1

        derived = derive_set(tcg, row.get("source", "") or "")
        if derived is None:
            still_blank_not_derivable[tcg] += 1
            continue
        if derived == "":
            still_blank_derivable[tcg] += 1
            continue
        row["set"] = derived
        backfilled[tcg] += 1
        changed += 1

    total_blank_before = sum(blank_before.values())

    print(f"Manifest: {len(rows)} rows, {total_blank_before} with a blank `set` before backfill.\n")
    print(f"{'tcg':<22}{'blank_before':<14}{'backfilled':<12}{'still_blank':<12}reason")
    print("-" * 90)
    for tcg in sorted(blank_before):
        n_backfilled = backfilled.get(tcg, 0)
        n_still = still_blank_derivable.get(tcg, 0) + still_blank_not_derivable.get(tcg, 0)
        if tcg in NOT_DERIVABLE_REASONS:
            reason = NOT_DERIVABLE_REASONS[tcg]
        elif tcg in DERIVABLE_TCGS:
            reason = "derivable; any still-blank rows had an id that didn't contain the expected separator"
        else:
            reason = "not threaded at download time (other_tcg catch-all / not_a_card / no card identity)"
        tcg_label = tcg or "(blank tcg)"
        print(f"{tcg_label:<22}{blank_before[tcg]:<14}{n_backfilled:<12}{n_still:<12}{reason}")

    print(f"\n{changed} row(s) backfilled. {total_blank_before - changed} row(s) remain blank.")

    if args.dry_run:
        print("\nDry run — manifest not written.")
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
