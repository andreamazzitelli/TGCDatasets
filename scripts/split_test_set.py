#!/usr/bin/env python3
"""Carve a held-out test set out of data/<model>/train/ and reconcile the manifest.

Create ML trains on data/<model>/train/<label>/ and, if present, reports a
final honest accuracy number against data/<model>/test/<label>/ (see
docs/createml_guide.md). This script:

1. Optionally caps each train class at --cap images, deleting the surplus
   (deterministic: files are sorted by name and the tail is dropped) so
   classes that were over-collected don't bias training.
2. Moves --test-per-class images per class from train/ to test/, chosen with
   a fixed --seed so the split is reproducible. Never leaves a train class
   below --min-train.
3. Rewrites data/manifest.csv: adds/updates the `split` column, drops rows
   whose file is no longer on disk, de-duplicates by filename, sorts.

It's idempotent-ish: files already in test/ stay there and count toward
--test-per-class, so re-running with the same numbers is a no-op.

    python scripts/split_test_set.py --test-per-class 60 --cap 360
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
MODELS = ["tcg_identifier", "grading_status"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".tiff"}
MANIFEST_HEADER = ["filename", "model", "split", "label", "tcg", "source", "license_note", "date_added", "notes"]


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-per-class", type=int, default=60, help="images to hold out per class (default: 60)")
    parser.add_argument("--cap", type=int, default=None, help="max images per train class; surplus is deleted")
    parser.add_argument("--min-train", type=int, default=120, help="never take a train class below this (default: 120)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print what would change, touch nothing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    moved_to_test: set[str] = set()
    deleted: set[str] = set()

    for model in MODELS:
        train_root = DATA_ROOT / model / "train"
        test_root = DATA_ROOT / model / "test"
        if not train_root.is_dir():
            continue
        for label_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
            label = label_dir.name
            train_imgs = images_in(label_dir)
            test_dir = test_root / label
            test_imgs = images_in(test_dir)

            # 1. cap
            if args.cap is not None and len(train_imgs) > args.cap:
                surplus = train_imgs[args.cap:]
                train_imgs = train_imgs[: args.cap]
                for p in surplus:
                    deleted.add(p.name)
                    print(f"  delete  {model}/{label}: {p.name}")
                    if not args.dry_run:
                        p.unlink()

            # 2. move to test
            have = len(test_imgs)
            want = max(0, args.test_per_class - have)
            takeable = max(0, len(train_imgs) - args.min_train)
            take = min(want, takeable)
            if want and take < want:
                print(f"  ! {model}/{label}: only moving {take}/{want} to test to keep >= {args.min_train} in train", file=sys.stderr)
            picks = rng.sample(train_imgs, take) if take else []
            for p in sorted(picks):
                dest = test_dir / p.name
                moved_to_test.add(p.name)
                print(f"  test    {model}/{label}: {p.name}")
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    p.rename(dest)

            final_train = len(train_imgs) - len(picks)
            final_test = have + len(picks)
            print(f"{model}/{label}: train={final_train} test={final_test}")

    reconcile_manifest(moved_to_test, deleted, dry_run=args.dry_run)


def reconcile_manifest(moved_to_test: set[str], deleted: set[str], *, dry_run: bool) -> None:
    if not MANIFEST_PATH.exists():
        return
    on_disk: dict[str, str] = {}  # filename -> split, from its actual location
    for model in MODELS:
        for split in ("train", "test"):
            root = DATA_ROOT / model / split
            if not root.is_dir():
                continue
            for label_dir in root.iterdir():
                if label_dir.is_dir():
                    for p in images_in(label_dir):
                        on_disk[p.name] = split

    with MANIFEST_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        fn = row["filename"]
        if fn in seen or fn not in on_disk:
            continue
        seen.add(fn)
        row["split"] = on_disk[fn]
        out.append({k: row.get(k, "") for k in MANIFEST_HEADER})
    out.sort(key=lambda r: (r["model"], r["label"], r["split"], r["filename"]))

    print(f"\nmanifest: {len(rows)} rows -> {len(out)} (dropped {len(rows) - len(out)} stale/dupe)")
    if dry_run:
        return
    with MANIFEST_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
