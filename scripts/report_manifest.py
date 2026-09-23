#!/usr/bin/env python3
"""Read-only summary report over data/manifest.csv + the files actually on
disk under data/ — the "brief §0 existing-assets audit" tooling described in
docs/tcg_detection_pipeline.md §2.

Covers: images per TCG/language/set, class imbalance (ratio to the largest
class), resolution range, file format breakdown, corrupted/unreadable
files, exact duplicates (content hash) and near-duplicates (perceptual
hash), and a photo_kind (catalog_scan vs. real_photo) breakdown.

`photo_kind` is derived at report time, not stored in the manifest (see
docs/tcg_detection_pipeline.md §2's "start derived-only" guidance): a row
is `real_photo` if its `source` column starts with the prefix the eBay
download scripts write ("ebay:" — see scripts/download/download_ebay_raw.py
and download_ebay_graded.py), else `catalog_scan`. This is a coarse proxy
(e.g. PSA cert photos from download_psa_certs.py are real slab photos too,
but don't start with "ebay:") — matches the design doc's own definition
exactly rather than inventing a broader one; revisit if that turns out to
matter.

Language isn't tracked anywhere in this repo's manifest or download
scripts today, so that axis is reported as "not currently tracked" rather
than invented.

Needs Pillow and imagehash (see scripts/download/requirements.txt):
    pip install -r scripts/download/requirements.txt

    python3 scripts/report_manifest.py                 # human-readable
    python3 scripts/report_manifest.py --json           # machine-readable
    python3 scripts/report_manifest.py --skip-hashing    # faster, no dup/near-dup pass
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None
    UnidentifiedImageError = Exception

try:
    import imagehash
except ImportError:
    imagehash = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
EBAY_SOURCE_PREFIX = "ebay:"  # see download_ebay_raw.py / download_ebay_graded.py
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".tiff", ".bmp", ".gif"}


def photo_kind(source: str) -> str:
    return "real_photo" if source.startswith(EBAY_SOURCE_PREFIX) else "catalog_scan"


def find_file(filename: str) -> Path | None:
    """Locate a manifest row's file under data/<model>/<split>/<label>/...
    by searching the two known model trees (this repo doesn't index files
    by filename anywhere else, and filenames aren't guaranteed unique
    across labels, so this returns the first match found).

    Also checks the shallower data/<model>/<label>/... shape used by
    scripts/augment.py's output (data/tcg_identifier_augmented/<label>/) —
    those aren't part of the Create ML train/test split tree, so they have
    no <split> segment (see docs/tcg_detection_pipeline.md §3.1)."""
    matches = list(DATA_ROOT.glob(f"*/*/*/{filename}"))
    if matches:
        return matches[0]
    matches = list(DATA_ROOT.glob(f"*/*/{filename}"))
    return matches[0] if matches else None


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"No manifest found at {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def build_report(rows: list[dict], *, skip_hashing: bool) -> dict:
    report: dict = {}

    # --- images per TCG / set, and per model+label (imbalance needs a class axis) ---
    by_tcg = Counter(r.get("tcg", "") or "(blank)" for r in rows)
    by_set = Counter(r.get("set", "") or "(blank)" for r in rows)
    by_class = Counter((r.get("model", ""), r.get("label", "")) for r in rows)

    report["total_rows"] = len(rows)
    report["images_per_tcg"] = dict(sorted(by_tcg.items(), key=lambda kv: -kv[1]))
    report["images_per_set"] = {
        "distinct_sets": len([k for k in by_set if k != "(blank)"]),
        "rows_with_no_set": by_set.get("(blank)", 0),
        "top_20_by_row_count": dict(Counter({k: v for k, v in by_set.items() if k != "(blank)"}).most_common(20)),
    }
    report["language"] = "not currently tracked — no language field exists in the manifest or any download_*.py script"

    # --- class imbalance: ratio of each (model, label) count to the largest in that model ---
    imbalance: dict[str, dict] = {}
    per_model_classes: dict[str, dict[str, int]] = defaultdict(dict)
    for (model, label), count in by_class.items():
        per_model_classes[model][label] = count
    for model, classes in per_model_classes.items():
        largest = max(classes.values()) if classes else 0
        imbalance[model] = {
            "largest_class_count": largest,
            "classes": {
                label: {"count": count, "ratio_to_largest": round(count / largest, 3) if largest else 0.0}
                for label, count in sorted(classes.items(), key=lambda kv: -kv[1])
            },
        }
    report["class_imbalance"] = imbalance

    # --- photo_kind (derived, not stored) ---
    kind_counts = Counter(photo_kind(r.get("source", "")) for r in rows)
    report["photo_kind"] = dict(kind_counts)

    # --- locate files on disk, resolution/format/corruption/hashing ---
    resolutions: list[tuple[int, int]] = []
    formats = Counter()
    corrupted: list[str] = []
    missing_on_disk: list[str] = []
    content_hashes: dict[str, list[str]] = defaultdict(list)
    phashes: dict[str, list[str]] = defaultdict(list)

    can_hash_images = Image is not None
    can_perceptual_hash = imagehash is not None

    for row in rows:
        filename = row.get("filename", "")
        path = find_file(filename)
        if path is None or not path.is_file():
            missing_on_disk.append(filename)
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            formats[suffix.lstrip(".")] += 1

        if not can_hash_images:
            continue

        if not skip_hashing:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            content_hashes[sha].append(filename)

        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:  # re-open: verify() leaves the file unusable for further ops
                resolutions.append(im.size)
                if can_perceptual_hash and not skip_hashing:
                    phashes[str(imagehash.phash(im))].append(filename)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupted.append(f"{filename}: {exc}")

    report["files_missing_on_disk"] = missing_on_disk
    report["file_format_breakdown"] = dict(formats.most_common())
    report["corrupted_or_unreadable"] = corrupted

    if resolutions:
        widths = [w for w, _ in resolutions]
        heights = [h for _, h in resolutions]
        report["resolution_range"] = {
            "min_width": min(widths), "max_width": max(widths),
            "min_height": min(heights), "max_height": max(heights),
            "images_measured": len(resolutions),
        }
    else:
        report["resolution_range"] = None

    if skip_hashing:
        report["exact_duplicates"] = "skipped (--skip-hashing)"
        report["near_duplicates"] = "skipped (--skip-hashing)"
    elif not can_hash_images:
        report["exact_duplicates"] = "skipped (Pillow not installed)"
        report["near_duplicates"] = "skipped (Pillow not installed)"
    else:
        exact_dupe_groups = {h: files for h, files in content_hashes.items() if len(files) > 1}
        report["exact_duplicates"] = {
            "duplicate_groups": len(exact_dupe_groups),
            "duplicate_files": sum(len(files) - 1 for files in exact_dupe_groups.values()),
            "groups": exact_dupe_groups,
        }
        if can_perceptual_hash:
            near_dupe_groups = {h: files for h, files in phashes.items() if len(files) > 1}
            report["near_duplicates"] = {
                "duplicate_groups": len(near_dupe_groups),
                "duplicate_files": sum(len(files) - 1 for files in near_dupe_groups.values()),
                "groups": near_dupe_groups,
            }
        else:
            report["near_duplicates"] = "skipped (imagehash not installed)"

    return report


def print_human(report: dict) -> None:
    print(f"Manifest rows: {report['total_rows']}")
    if report["files_missing_on_disk"]:
        print(f"  ! {len(report['files_missing_on_disk'])} manifest row(s) have no file on disk (showing up to 10):")
        for f in report["files_missing_on_disk"][:10]:
            print(f"      {f}")

    print("\n=== Images per TCG ===")
    for tcg, count in report["images_per_tcg"].items():
        print(f"  {tcg:<24} {count}")

    print("\n=== Language ===")
    print(f"  {report['language']}")

    print("\n=== Images per set ===")
    s = report["images_per_set"]
    print(f"  distinct sets: {s['distinct_sets']}, rows with no recoverable set: {s['rows_with_no_set']}")
    if s["top_20_by_row_count"]:
        print("  top sets by row count:")
        for set_id, count in s["top_20_by_row_count"].items():
            print(f"    {set_id:<24} {count}")

    print("\n=== Class imbalance (ratio to largest class in the model) ===")
    for model, info in report["class_imbalance"].items():
        print(f"  {model} (largest class = {info['largest_class_count']}):")
        for label, stats in info["classes"].items():
            flag = "  <-- imbalanced (<0.5x)" if stats["ratio_to_largest"] < 0.5 else ""
            print(f"    {label:<20} {stats['count']:<6} ratio={stats['ratio_to_largest']:.2f}{flag}")

    print("\n=== Resolution range ===")
    r = report["resolution_range"]
    if r:
        print(f"  width:  {r['min_width']}-{r['max_width']}px")
        print(f"  height: {r['min_height']}-{r['max_height']}px")
        print(f"  ({r['images_measured']} images measured)")
    else:
        print("  no images measured (Pillow missing, or no files found on disk)")

    print("\n=== File format breakdown ===")
    for fmt, count in report["file_format_breakdown"].items():
        print(f"  {fmt:<8} {count}")

    print("\n=== Corrupted / unreadable files ===")
    if isinstance(report["corrupted_or_unreadable"], list):
        if report["corrupted_or_unreadable"]:
            for c in report["corrupted_or_unreadable"]:
                print(f"  ! {c}")
        else:
            print("  none found")

    print("\n=== Exact duplicates (content hash) ===")
    ed = report["exact_duplicates"]
    if isinstance(ed, dict):
        print(f"  {ed['duplicate_groups']} duplicate group(s), {ed['duplicate_files']} redundant file(s)")
        for h, files in list(ed["groups"].items())[:10]:
            print(f"    {h[:12]}...: {files}")
    else:
        print(f"  {ed}")

    print("\n=== Near duplicates (perceptual hash) ===")
    nd = report["near_duplicates"]
    if isinstance(nd, dict):
        print(f"  {nd['duplicate_groups']} duplicate group(s), {nd['duplicate_files']} redundant file(s)")
        for h, files in list(nd["groups"].items())[:10]:
            print(f"    {h}: {files}")
    else:
        print(f"  {nd}")

    print("\n=== photo_kind (derived from `source`, not a stored column) ===")
    for kind, count in report["photo_kind"].items():
        print(f"  {kind:<14} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output instead of the human-readable report")
    parser.add_argument("--skip-hashing", action="store_true", help="skip the content-hash/perceptual-hash pass (faster, no dup/near-dup section)")
    args = parser.parse_args()

    rows = load_manifest()
    report = build_report(rows, skip_hashing=args.skip_hashing)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
