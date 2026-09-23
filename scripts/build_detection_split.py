#!/usr/bin/env python3
"""Assign every row in data/manifest.csv to a train/index/eval bucket, by
SET (not by image), for the new on-device TCG-detection pipeline.

See docs/tcg_detection_pipeline.md §2. This is deliberately separate from
scripts/split_test_set.py, which keeps meaning "which data/<model>/<split>/
folder is this file physically in" for the Create ML workflow — that split
is by image and stays untouched. This script writes a new `detection_split`
column instead, so the two concepts never collide in the manifest.

Why by set, not by image: the new pipeline's eval set must measure
generalization to *unseen* cards. If two images of the same card (or two
prints from the same set) can land in both train and eval, eval accuracy is
inflated by memorization rather than generalization. So the unit of
assignment here is a "set" — everything sharing one `set` value moves
together.

Rows with a blank `set` column (self-photographed images, eBay listings,
sources where no set identifier was recoverable at download time — see
scripts/download/common.py and docs/tcg_detection_pipeline.md §2) do NOT
get pooled into one shared bucket together. Each such row is treated as its
own singleton "set", keyed by its filename. Silently merging them would let
otherwise-unrelated real photos leak across train/index/eval as if they
were "the same set" of images, and (for eBay in particular) would make eval
accuracy misleadingly easy or hard depending on which bucket that giant
pseudo-set landed in.

Split proportions: default 70% train / 15% index / 15% eval, by SET count
(not image count) within each (model, label) group. These are reasonable
defaults for a small curated dataset, not derived from the design doc (it
left exact proportions unspecified) — tune with --train-frac/--eval-frac as
the real per-class set counts become clear (index gets the remainder).
Assignment is a deterministic hash-bucket keyed on (model, label, set) and
--seed, so re-running with the same seed and manifest is a no-op, and
adding new sets to a class doesn't reshuffle the sets already assigned.

    python3 scripts/build_detection_split.py --dry-run
    python3 scripts/build_detection_split.py --train-frac 0.7 --eval-frac 0.15
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
BUCKETS = ("train", "index", "eval")


def set_key(row: dict) -> str:
    """The grouping key for a manifest row: its `set` value if present,
    else a singleton key derived from its own filename so blank-set rows
    never pool together (see module docstring)."""
    set_value = (row.get("set") or "").strip()
    if set_value:
        return set_value
    return f"__singleton__:{row.get('filename', '')}"


def assign_bucket(model: str, label: str, key: str, seed: int, train_frac: float, eval_frac: float) -> str:
    """Deterministic hash-bucket assignment for one (model, label, set) key.

    Hashing (rather than shuffling a list) means the bucket for a given
    key depends only on (model, label, key, seed) — not on how many other
    sets currently exist in that class — so re-runs are stable and adding a
    brand-new set later doesn't reshuffle sets already assigned.
    """
    digest = hashlib.sha256(f"{seed}:{model}:{label}:{key}".encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    if frac < train_frac:
        return "train"
    if frac < train_frac + eval_frac:
        return "eval"
    return "index"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="hash seed for reproducible assignment (default: 42)")
    parser.add_argument("--train-frac", type=float, default=0.70, help="fraction of sets assigned to train (default: 0.70)")
    parser.add_argument("--eval-frac", type=float, default=0.15, help="fraction of sets assigned to eval (default: 0.15); index gets the remainder")
    parser.add_argument("--dry-run", action="store_true", help="report what would be assigned, write nothing")
    args = parser.parse_args()

    if args.train_frac < 0 or args.eval_frac < 0 or args.train_frac + args.eval_frac > 1:
        parser.error("--train-frac and --eval-frac must be >= 0 and sum to <= 1 (index gets the remainder)")

    if not MANIFEST_PATH.exists():
        parser.error(f"No manifest found at {MANIFEST_PATH}")

    with MANIFEST_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "set" not in fieldnames:
        parser.error("manifest has no `set` column yet — run the migrated scripts/download/common.py first")

    # 1. Group rows by (model, label, set-key).
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "")
        label = row.get("label", "")
        key = set_key(row)
        groups[(model, label, key)].append(row)

    # 2. Assign each group's bucket once, deterministically.
    group_bucket: dict[tuple[str, str, str], str] = {}
    for (model, label, key) in groups:
        group_bucket[(model, label, key)] = assign_bucket(model, label, key, args.seed, args.train_frac, args.eval_frac)

    # 3. Apply to rows and write back `detection_split` (new column,
    #    distinct from the existing `split` column).
    out_fieldnames = fieldnames if "detection_split" in fieldnames else fieldnames + ["detection_split"]
    changed = 0
    singleton_rows = 0
    for row in rows:
        model = row.get("model", "")
        label = row.get("label", "")
        key = set_key(row)
        if key.startswith("__singleton__:"):
            singleton_rows += 1
        bucket = group_bucket[(model, label, key)]
        if row.get("detection_split") != bucket:
            changed += 1
        row["detection_split"] = bucket

    # 4. Summary report.
    per_class_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {b: 0 for b in BUCKETS})
    per_class_sets: dict[tuple[str, str], dict[str, set]] = defaultdict(lambda: {b: set() for b in BUCKETS})
    for row in rows:
        model, label = row.get("model", ""), row.get("label", "")
        bucket = row["detection_split"]
        per_class_counts[(model, label)][bucket] += 1
        per_class_sets[(model, label)][bucket].add(set_key(row))

    print(f"Manifest: {len(rows)} rows, {len(groups)} distinct sets ({singleton_rows} rows with no recoverable set, "
          f"each its own singleton).")
    print(f"Target proportions: train={args.train_frac:.0%} eval={args.eval_frac:.0%} "
          f"index={1 - args.train_frac - args.eval_frac:.0%} (by set count, seed={args.seed})\n")
    header = f"{'model':<16}{'label':<22}{'train (sets/imgs)':<20}{'index (sets/imgs)':<20}{'eval (sets/imgs)':<20}"
    print(header)
    print("-" * len(header))
    for (model, label) in sorted(per_class_counts):
        counts = per_class_counts[(model, label)]
        sets = per_class_sets[(model, label)]
        cells = "".join(f"{len(sets[b])}/{counts[b]:<17}" for b in BUCKETS)
        print(f"{model:<16}{label:<22}{cells}")

    print(f"\n{changed} row(s) would change `detection_split`." if args.dry_run else f"\n{changed} row(s) had their `detection_split` set/changed.")

    if args.dry_run:
        print("Dry run — manifest not written.")
        return

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in out_fieldnames})
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
