#!/usr/bin/env python3
"""Camera-realism simulation: turn clean catalog scans into augmented copies
that look more like a real phone-camera photo of a card (glare, perspective,
blur, JPEG artifacts, sleeve reflections) — see
docs/tcg_detection_pipeline.md §3.1.

Why this exists (not the same as Create ML's built-in augmentation panel,
docs/createml_guide.md §2): Create ML's Crop/Rotate/Blur/Expose/Noise panel
only perturbs the *scan* distribution at training time. It has no concept of
sleeve glare, perspective distortion from a hand-held phone, or JPEG
recompression — exactly the gap between "clean catalog scan" and "photo taken
through a sleeve on a table" that the on-device classifier has to bridge.

Inputs: rows in data/manifest.csv that are
  - `photo_kind == catalog_scan` (the same derivation report_manifest.py
    uses: everything except an `ebay:`-prefixed `source` — see
    scripts/report_manifest.py's `photo_kind()`, imported here rather than
    reimplemented so the two scripts can never quietly disagree on the
    definition), and
  - `detection_split` in {train, index} — eval must stay real and
    unaugmented (the eval-set-integrity rule established by
    scripts/build_detection_split.py: augmenting eval would let the
    classifier be graded on inputs derived from images it may have trained
    on, and defeats the "held-out unseen cards" point of eval entirely).

Output: augmented copies under data/tcg_identifier_augmented/<label>/ — a
new top-level tree, deliberately not nested under data/tcg_identifier/ and
not split into train/test subfolders (augmented images aren't part of the
Create ML train/test split; `split_test_set.py` never sees them). Symlinked
to the external SSD exactly like data/tcg_identifier/ already is (see
README.md "why images aren't committed here"), so this repo's git history
never grows.

Each augmented image is logged to data/manifest.csv via
scripts/download/common.py's `append_manifest`:
  - `source`      = f"augment:{original_filename}" (traces back to the
                     exact source row; the original's own `source` string
                     is preserved in `notes` too, for full provenance)
  - `notes`        = which transforms were applied + the original `source`
  - `model`/`tcg`/`label`/`set` = copied unchanged from the source row
                     (augmentation doesn't change what card/set/TCG this is)
  - `detection_split` = copied unchanged from the source row (an augmented
                     copy of a `train` image is itself `train`; same for
                     `index`; `eval` rows are never read as input at all,
                     so this can never produce an `eval` augmented row)
  - `split`        = "train" (a nominal default; these files aren't under
                     any data/<model>/<split>/ folder, so the Create-ML
                     "physical folder" meaning of this column doesn't
                     really apply to them — see split_test_set.py)

Transforms (Pillow only — ImageFilter/ImageEnhance/Image.transform/
ImageDraw; glare, sleeve-overlay and partial-shadow are procedurally drawn,
no external texture asset required):
  perspective warp, gaussian blur, glare/specular highlight overlay,
  color/brightness/contrast jitter, sleeve overlay (diagonal highlight band
  + slight brightness/contrast lift), JPEG re-encode at a lossy quality,
  partial soft shadow.

A random subset of 2-4 of these seven is applied per output image (not all
seven — that would look uniform and unrealistic), and --variants-per-image
copies are produced per source image, all seeded by --seed for
reproducibility.

    python3 scripts/augment.py --dry-run --limit 20
    python3 scripts/augment.py --variants-per-image 3
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "download"))
from download.common import REPO_ROOT, append_manifest  # noqa: E402
from report_manifest import find_file, photo_kind  # noqa: E402

DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
OUT_ROOT = DATA_ROOT / "tcg_identifier_augmented"
MODEL = "tcg_identifier"  # the only model this repo augments for right now
ELIGIBLE_SPLITS = {"train", "index"}
TRANSFORM_NAMES = ["perspective", "blur", "glare", "jitter", "sleeve", "jpeg", "shadow"]


# --- pure-Python perspective coefficient solve (no numpy dependency) ---

def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a square system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular matrix in perspective coefficient solve")
        m[col], m[pivot] = m[pivot], m[col]
        pivot_val = m[col][col]
        m[col] = [v / pivot_val for v in m[col]]
        for r in range(n):
            if r != col:
                factor = m[r][col]
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]
    return [row[-1] for row in m]


def find_perspective_coeffs(source_quad: list[tuple[float, float]], target_rect: list[tuple[float, float]]) -> list[float]:
    """Coefficients for Image.transform(..., Image.PERSPECTIVE, coeffs) that
    map `target_rect` (the plain output rectangle corners) back to
    `source_quad` (where those corners come from in the input image)."""
    matrix = []
    for (sx, sy), (tx, ty) in zip(source_quad, target_rect):
        matrix.append([tx, ty, 1, 0, 0, 0, -tx * sx, -ty * sx])
        matrix.append([0, 0, 0, tx, ty, 1, -tx * sy, -ty * sy])
    b = [coord for corner in source_quad for coord in corner]
    return _solve_linear_system(matrix, b)


# --- transforms (each takes (image, rng) and returns a new image) ---

def t_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    jitter = 0.06  # max corner displacement as a fraction of width/height
    rect = [(0, 0), (w, 0), (w, h), (0, h)]
    quad = [
        (x + rng.uniform(-jitter, jitter) * w, y + rng.uniform(-jitter, jitter) * h)
        for x, y in rect
    ]
    coeffs = find_perspective_coeffs(quad, rect)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def t_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.8)))


def t_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    ew, eh = rng.uniform(0.2, 0.45) * w, rng.uniform(0.12, 0.3) * h
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    draw.ellipse([cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2], fill=255)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(ew, eh) * 0.25))
    opacity = rng.uniform(0.25, 0.55)
    white = Image.new("RGB", (w, h), (255, 255, 255))
    alpha = overlay.point(lambda v: int(v * opacity))
    return Image.composite(white, img, alpha)


def t_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    out = ImageEnhance.Color(img).enhance(rng.uniform(0.75, 1.25))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.25))
    return out


def t_sleeve(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    # Slight overall lift to simulate looking through plastic.
    out = ImageEnhance.Brightness(img).enhance(rng.uniform(1.03, 1.12))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.9, 0.98))
    # A soft diagonal highlight band.
    band = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(band)
    band_w = rng.uniform(0.15, 0.3) * max(w, h)
    offset = rng.uniform(-0.3, 1.3) * (w + h)
    x0, y0 = -w, offset - band_w
    x1, y1 = w * 2, offset + band_w
    draw.polygon([(x0, offset - band_w), (x1, offset - band_w + (x1 - x0)),
                  (x1, offset + band_w + (x1 - x0)), (x0, offset + band_w)], fill=180)
    band = band.rotate(rng.uniform(20, 50), resample=Image.BICUBIC)
    band = band.resize((w, h))
    band = band.filter(ImageFilter.GaussianBlur(radius=band_w * 0.3))
    opacity = rng.uniform(0.12, 0.3)
    alpha = band.point(lambda v: int(v * opacity))
    white = Image.new("RGB", (w, h), (255, 255, 255))
    return Image.composite(white, out, alpha)


def t_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(35, 70)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert(img.mode if img.mode in ("RGB", "L") else "RGB").copy()


def t_shadow(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    side = rng.choice(["left", "right", "top", "bottom", "corner"])
    frac = rng.uniform(0.25, 0.55)
    if side == "left":
        box = [0, 0, w * frac, h]
    elif side == "right":
        box = [w * (1 - frac), 0, w, h]
    elif side == "top":
        box = [0, 0, w, h * frac]
    elif side == "bottom":
        box = [0, h * (1 - frac), w, h]
    else:
        box = [0, 0, w * frac, h * frac]
    draw.rectangle(box, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=min(w, h) * 0.12))
    opacity = rng.uniform(0.2, 0.45)
    alpha = mask.point(lambda v: int(v * opacity))
    black = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(black, img, alpha)


TRANSFORMS = {
    "perspective": t_perspective,
    "blur": t_blur,
    "glare": t_glare,
    "jitter": t_jitter,
    "sleeve": t_sleeve,
    "jpeg": t_jpeg,
    "shadow": t_shadow,
}


def choose_transforms(rng: random.Random) -> list[str]:
    k = rng.randint(2, 4)
    return rng.sample(TRANSFORM_NAMES, k)


def apply_transforms(img: Image.Image, names: list[str], rng: random.Random) -> Image.Image:
    out = img
    # Apply in the canonical TRANSFORM_NAMES order regardless of the random
    # subset's own ordering, so results are reproducible per (source,
    # variant, seed) and independent of set-iteration order.
    for name in TRANSFORM_NAMES:
        if name in names:
            out = TRANSFORMS[name](out, rng)
    return out


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"No manifest found at {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def eligible_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("model", "") != MODEL:
            continue
        if photo_kind(row.get("source", "")) != "catalog_scan":
            continue
        if row.get("detection_split", "") not in ELIGIBLE_SPLITS:
            continue
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for transform choice/params (default: 42)")
    parser.add_argument("--variants-per-image", type=int, default=3, help="augmented copies per source image (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of SOURCE images processed (for smoke-testing)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be generated, write nothing")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="quality for the final save of non-'jpeg'-transform outputs (default: 90)")
    args = parser.parse_args()

    rows = load_manifest()
    candidates = eligible_rows(rows)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    print(f"Manifest: {len(rows)} rows total. Eligible source images (model={MODEL}, "
          f"photo_kind=catalog_scan, detection_split in {sorted(ELIGIBLE_SPLITS)}): {len(candidates)}"
          + (f" (capped to {args.limit})" if args.limit is not None else "") + ".")

    if not candidates:
        print("Nothing to do.")
        return

    rng = random.Random(args.seed)
    produced = 0
    missing = 0
    by_split_produced = {"train": 0, "index": 0}

    for row in candidates:
        filename = row.get("filename", "")
        src_path = find_file(filename)
        if src_path is None or not src_path.is_file():
            missing += 1
            print(f"  ! source file not found on disk, skipping: {filename}", file=sys.stderr)
            continue

        label = row.get("label", "")
        tcg = row.get("tcg", "")
        set_value = row.get("set", "")
        detection_split = row.get("detection_split", "")
        orig_source = row.get("source", "")
        stem = Path(filename).stem
        out_dir = OUT_ROOT / label

        for variant_idx in range(1, args.variants_per_image + 1):
            # Sub-seed per (row, variant) so re-running the whole script
            # with the same --seed reproduces identical outputs regardless
            # of candidate-list order or --limit slicing changes upstream.
            variant_rng = random.Random(f"{args.seed}:{filename}:{variant_idx}")
            names = choose_transforms(variant_rng)
            abbrev = "-".join(n[:5] for n in names)
            out_filename = f"{stem}__aug{variant_idx:02d}_{abbrev}.jpg"
            out_path = out_dir / out_filename

            if args.dry_run:
                print(f"  [dry-run] {filename} -> {out_path.relative_to(REPO_ROOT)} "
                      f"(transforms={names}, detection_split={detection_split})")
                produced += 1
                by_split_produced[detection_split] = by_split_produced.get(detection_split, 0) + 1
                continue

            try:
                with Image.open(src_path) as im:
                    im = im.convert("RGB")
                    out_img = apply_transforms(im, names, variant_rng)
            except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the whole run
                print(f"  ! failed to augment {filename}: {exc}", file=sys.stderr)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            save_quality = args.jpeg_quality if "jpeg" not in names else rng.randint(35, 70)
            out_img.convert("RGB").save(out_path, format="JPEG", quality=save_quality)

            append_manifest(
                filename=out_filename,
                model=MODEL,
                label=label,
                tcg=tcg,
                source=f"augment:{filename}",
                license_note=row.get("license_note", ""),
                notes=f"transforms={','.join(names)}; orig_source={orig_source}",
                split="train",
                set=set_value,
                detection_split=detection_split,
            )
            produced += 1
            by_split_produced[detection_split] = by_split_produced.get(detection_split, 0) + 1

    verb = "Would produce" if args.dry_run else "Produced"
    print(f"\n{verb} {produced} augmented image(s) from {len(candidates) - missing} source image(s) "
          f"({missing} source file(s) missing on disk, skipped).")
    print(f"By detection_split: {by_split_produced}")
    if args.dry_run:
        print("Dry run — no files written, manifest not touched.")


if __name__ == "__main__":
    main()
