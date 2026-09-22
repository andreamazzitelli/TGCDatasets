#!/usr/bin/env python3
"""Self-contained DINOv2 embedding job, meant to run on a remote CPU box
(TCGInfrastructure's OCI VM for this pipeline) — see
docs/tcg_detection_pipeline.md §3.2 and scripts/build_index.py.

Not imported by anything else in this repo; it's transferred to the remote
host on its own (scp) alongside a staging dir built by
`build_index.py prepare` (images/<tcg>/<filename> + metadata.json) and run
there directly:

    python3 _remote_embed.py --images-dir images --metadata metadata.json \
        --out embeddings.npz --batch-size 4

Mirrors TCG_ETL's own embedding convention (`transform/embeddings.py`) for
consistency — same checkpoint, CLS token from `last_hidden_state[:, 0, :]`,
L2-normalized — even though this is a separate embedding pass, not a reuse
of TCG_ETL's export (TCG_ETL's DB has no precomputed embeddings to reuse).

Kept dependency-light and torch/transformers imports lazy so `--help` and
argument errors don't require them installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DINOV2_MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIMENSION = 384


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, help="directory containing <tcg>/<filename> images (from build_index.py prepare)")
    parser.add_argument("--metadata", required=True, help="metadata.json from build_index.py prepare")
    parser.add_argument("--out", required=True, help="output .npz path (filename/rel_path -> 384-dim float32 vector)")
    parser.add_argument("--batch-size", type=int, default=4, help="keep small on a memory-constrained box (default: 4)")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU thread cap (default: 2, to leave headroom for other processes on the box)")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(max(1, args.threads))

    images_dir = Path(args.images_dir)
    metadata = json.loads(Path(args.metadata).read_text())
    print(f"{len(metadata)} images listed in metadata", flush=True)

    print(f"loading {DINOV2_MODEL_NAME} (CPU)...", flush=True)
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
    model = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    model.eval()
    hidden = getattr(model.config, "hidden_size", None)
    if hidden != EMBEDDING_DIMENSION:
        sys.exit(f"unexpected hidden size {hidden}, expected {EMBEDDING_DIMENSION}")
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    results: dict[str, "np.ndarray"] = {}
    failed: list[str] = []

    batch: list[tuple[str, "Image.Image"]] = []

    def flush_batch() -> None:
        if not batch:
            return
        keys = [k for k, _ in batch]
        images = [img for _, img in batch]
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors="pt")
            features = model(**inputs).last_hidden_state[:, 0, :]
            features = F.normalize(features.float(), p=2, dim=1)
        vectors = features.cpu().numpy().astype(np.float32)
        for key, vec in zip(keys, vectors):
            results[key] = vec
        batch.clear()

    t_start = time.time()
    for i, row in enumerate(metadata):
        rel_path = row["rel_path"]
        img_path = images_dir / rel_path
        if not img_path.exists():
            failed.append(rel_path)
            continue
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB").copy()
        except Exception as exc:  # noqa: BLE001 - want to keep going, log and skip
            print(f"  ! failed to open {rel_path}: {exc}", file=sys.stderr)
            failed.append(rel_path)
            continue
        batch.append((rel_path, rgb))
        if len(batch) >= args.batch_size:
            flush_batch()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  {i + 1}/{len(metadata)} images processed ({elapsed:.0f}s elapsed)", flush=True)
    flush_batch()

    elapsed = time.time() - t_start
    print(f"embedded {len(results)}/{len(metadata)} images in {elapsed:.0f}s ({len(failed)} failed)", flush=True)
    if failed:
        print(f"failed rel_paths (first 10): {failed[:10]}", file=sys.stderr)

    np.savez(args.out, **results)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
