#!/usr/bin/env python3
"""On-device prototype index builder (signal A) — see
docs/tcg_detection_pipeline.md §3.2.

TCG_ETL's own database has zero pre-computed embeddings (confirmed
separately — the export pipeline described in
`TCG_ETL/docs/embedding-export-for-tcg-detection.md` was never run against
real data), so the "reuse TCG_ETL's embeddings" path in §3.2 is not
available. Instead this script embeds this repo's *own* index-split
reference images directly, with the same checkpoint TCG_ETL uses server-side
(`facebook/dinov2-small`, 384-dim — confirmed via `TCG_ETL/config/settings.py`
`DINOV2_MODEL_NAME`/`EMBEDDING_DIMENSION`).

The heavy DINOv2 inference is CPU-bound and this repo has no GPU box of its
own; it runs on TCGInfrastructure's already-provisioned OCI VM instead of
locally, so this script is deliberately split into two phases that don't
share a process:

  prepare   — selects the index-split images for the target TCGs from
              data/manifest.csv, copies them into a flat staging tree plus a
              metadata.json sidecar, ready to scp to the VM alongside
              scripts/_remote_embed.py.
  finalize  — takes back a results file (.npz, filename -> 384-dim vector)
              produced by scripts/_remote_embed.py on the VM, k-means
              clusters each TCG's vectors down to --prototypes-per-tcg
              prototypes (clustering ~1-2k vectors is cheap — this stays
              local, no need to run it remotely), and writes the compact
              binary index consumed later by the iOS app.

Usage:

    python scripts/build_index.py prepare --staging-dir /tmp/tcg_index_staging
    # ... transfer staging-dir + scripts/_remote_embed.py to the VM, run it,
    # transfer the resulting embeddings.npz back ...
    python scripts/build_index.py finalize \
        --staging-dir /tmp/tcg_index_staging \
        --embeddings /tmp/tcg_index_staging/embeddings.npz \
        --prototypes-per-tcg 32
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
TCG_IDENTIFIER_DIR = DATA_ROOT / "tcg_identifier"
TCG_IDENTIFIER_AUGMENTED_DIR = DATA_ROOT / "tcg_identifier_augmented"
INDEX_OUT_DIR = DATA_ROOT / "index"

# The 4 TCGs decided with the user for this build (not the full
# docs/taxonomy.md list — the design doc's "few dozen prototypes per TCG"
# scope was narrowed to these 4 for this pass; re-running with more TCGs
# added to this tuple is the intended way to extend it later).
TARGET_TCGS = ("pokemon", "yugioh", "magic_the_gathering", "one_piece")

DINOV2_MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIMENSION = 384


def _is_augmented(row: dict[str, str]) -> bool:
    # Mirrors scripts/augment.py's own provenance convention: augmented
    # rows carry `source=augment:<original_filename>` and land physically
    # under data/tcg_identifier_augmented/<label>/, not
    # data/tcg_identifier/<split>/<label>/ like everything else.
    return row["source"].startswith("augment:") or "__aug" in row["filename"]


def _resolve_image_path(row: dict[str, str]) -> Path:
    if _is_augmented(row):
        return TCG_IDENTIFIER_AUGMENTED_DIR / row["label"] / row["filename"]
    return TCG_IDENTIFIER_DIR / row["split"] / row["label"] / row["filename"]


def _read_index_rows(tcgs: tuple[str, ...]) -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found: {MANIFEST_PATH}")
    rows = []
    with MANIFEST_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != "tcg_identifier":
                continue
            if row.get("detection_split") != "index":
                continue
            if row.get("tcg") not in tcgs:
                continue
            rows.append(row)
    return rows


def cmd_prepare(args: argparse.Namespace) -> None:
    tcgs = tuple(args.tcgs)
    rows = _read_index_rows(tcgs)
    if not rows:
        sys.exit("no index-split rows found for the requested TCGs — check data/manifest.csv")

    staging_dir = Path(args.staging_dir).resolve()
    images_dir = staging_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, Any]] = []
    missing: list[str] = []
    by_tcg: dict[str, int] = defaultdict(int)
    total_bytes = 0

    for row in rows:
        src = _resolve_image_path(row)
        if not src.exists():
            missing.append(str(src))
            continue
        tcg = row["tcg"]
        dest_dir = images_dir / tcg
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / row["filename"]
        if not dest.exists():
            shutil.copyfile(src, dest)
        total_bytes += dest.stat().st_size
        by_tcg[tcg] += 1
        metadata.append(
            {
                "rel_path": f"{tcg}/{row['filename']}",
                "filename": row["filename"],
                "tcg": tcg,
                "label": row["label"],
                "set": row.get("set", ""),
                "source": row["source"],
                "augmented": _is_augmented(row),
            }
        )

    (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"staged {len(metadata)} images ({total_bytes / 1e6:.1f} MB) into {images_dir}")
    for tcg in tcgs:
        print(f"  {tcg}: {by_tcg.get(tcg, 0)}")
    if missing:
        print(f"WARNING: {len(missing)} manifest rows had no file on disk, skipped:", file=sys.stderr)
        for m in missing[:10]:
            print(f"  missing: {m}", file=sys.stderr)
    print(f"\nnext: copy {staging_dir} and scripts/_remote_embed.py to the remote host, run:")
    print(f"  python3 _remote_embed.py --images-dir images --metadata metadata.json --out embeddings.npz")
    print("then copy embeddings.npz back and run this script's `finalize` phase.")


def _kmeans(vectors: "Any", k: int, *, seed: int) -> tuple["Any", "Any"]:
    """Cluster `vectors` (n, dim) into k prototypes. Prefers scikit-learn
    (already a common dependency in this ecosystem) but falls back to a
    plain numpy Lloyd's-algorithm implementation so this script doesn't
    force a new heavy dependency just for clustering ~1-2k vectors."""
    import numpy as np

    n = vectors.shape[0]
    k = max(1, min(k, n))

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        assignments = km.fit_predict(vectors)
        centers = km.cluster_centers_
        return centers, assignments
    except ImportError:
        pass

    # Minimal numpy k-means++ fallback.
    rng = np.random.default_rng(seed)
    centers = np.empty((k, vectors.shape[1]), dtype=vectors.dtype)
    centers[0] = vectors[rng.integers(0, n)]
    for i in range(1, k):
        d2 = np.min(((vectors[:, None, :] - centers[None, :i, :]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers[i] = vectors[rng.choice(n, p=probs)]

    assignments = np.zeros(n, dtype=int)
    for _ in range(100):
        dists = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_assignments = dists.argmin(axis=1)
        if np.array_equal(new_assignments, assignments) and _ > 0:
            assignments = new_assignments
            break
        assignments = new_assignments
        for c in range(k):
            members = vectors[assignments == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return centers, assignments


def cmd_finalize(args: argparse.Namespace) -> None:
    import numpy as np

    staging_dir = Path(args.staging_dir).resolve()
    metadata = json.loads((staging_dir / "metadata.json").read_text())
    embeddings_path = Path(args.embeddings).resolve()
    if not embeddings_path.exists():
        sys.exit(f"embeddings file not found: {embeddings_path}")

    npz = np.load(embeddings_path)
    keys = set(npz.files)

    by_tcg: dict[str, list["Any"]] = defaultdict(list)
    missing_embeddings = 0
    for row in metadata:
        key = row["rel_path"]
        if key not in keys:
            missing_embeddings += 1
            continue
        vec = npz[key].astype(np.float32)
        by_tcg[row["tcg"]].append(vec)

    if missing_embeddings:
        print(f"WARNING: {missing_embeddings} staged images have no matching embedding", file=sys.stderr)

    all_rows: list[dict[str, Any]] = []
    all_vectors: list["Any"] = []

    for tcg in sorted(by_tcg):
        vectors = np.stack(by_tcg[tcg])
        if vectors.shape[1] != EMBEDDING_DIMENSION:
            sys.exit(f"{tcg}: expected {EMBEDDING_DIMENSION}-dim vectors, got {vectors.shape[1]}")
        k = args.prototypes_per_tcg
        centers, assignments = _kmeans(vectors, k, seed=args.seed)
        counts = np.bincount(assignments, minlength=centers.shape[0])
        for i, center in enumerate(centers):
            all_rows.append(
                {
                    "tcg": tcg,
                    "prototype_index": i,
                    "cluster_size": int(counts[i]),
                }
            )
            all_vectors.append(center)
        print(f"{tcg}: {vectors.shape[0]} embeddings -> {centers.shape[0]} prototypes")

    matrix = np.stack(all_vectors).astype(np.float16)

    INDEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = INDEX_OUT_DIR / "prototypes.f16.bin"
    labels_path = INDEX_OUT_DIR / "labels.json"

    matrix.tofile(bin_path)
    labels_doc = {
        "model": DINOV2_MODEL_NAME,
        "dim": EMBEDDING_DIMENSION,
        "dtype": "float16",
        "count": matrix.shape[0],
        "prototypes_per_tcg": args.prototypes_per_tcg,
        "tcgs": sorted(by_tcg),
        "rows": all_rows,
    }
    labels_path.write_text(json.dumps(labels_doc, indent=2))

    print(f"\nwrote {bin_path} ({bin_path.stat().st_size} bytes)")
    print(f"wrote {labels_path} ({labels_path.stat().st_size} bytes)")
    print(f"total prototypes: {matrix.shape[0]} across {len(by_tcg)} TCGs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    p_prepare = sub.add_parser("prepare", help="stage index-split images + metadata for remote embedding")
    p_prepare.add_argument("--staging-dir", required=True, help="local directory to write images/ + metadata.json into")
    p_prepare.add_argument("--tcgs", nargs="+", default=list(TARGET_TCGS), help="TCGs to include (default: the 4 target TCGs)")
    p_prepare.set_defaults(func=cmd_prepare)

    p_finalize = sub.add_parser("finalize", help="cluster returned embeddings and write the prototype index")
    p_finalize.add_argument("--staging-dir", required=True, help="the same directory passed to `prepare`")
    p_finalize.add_argument("--embeddings", required=True, help="path to the embeddings.npz produced by _remote_embed.py")
    p_finalize.add_argument("--prototypes-per-tcg", type=int, default=32, help="k-means cluster count per TCG (default: 32)")
    p_finalize.add_argument("--seed", type=int, default=0)
    p_finalize.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
