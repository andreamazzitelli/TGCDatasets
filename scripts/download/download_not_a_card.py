#!/usr/bin/env python3
"""Populate data/tcg_identifier/train/not_a_card/ with real-world photos that
are NOT trading cards, from Lorem Picsum (https://picsum.photos).

This is the identifier model's negative class: what the classifier should
fall back to when the phone camera isn't pointed at a card at all (a desk, a
hand, a room, an object). Picsum serves photos from the Unsplash catalog
(Unsplash License — free to use, no permission needed), which gives a broad
mix of scenes/objects/people without any card-like framing.

  GET /v2/list?page=<n>&limit=<k>  -> [{id, author, url, download_url}, ...]

Images are pulled at a portrait-ish size close to how a card photo is
framed, so the negatives sit in a similar aspect range to the positives.

NOTE: Picsum's catalog is ~1000 images, so --limit above that will just
fetch everything available. For a bigger/720p-varied negative set, also drop
in your own photos of non-card subjects (see docs/data_collection.md).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REPO_ROOT,
    append_manifest,
    download_image,
    make_session,
    sleep_polite,
)

LABEL = "not_a_card"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
LIST_URL = "https://picsum.photos/v2/list"
# Portrait crop roughly matching a hand-held card photo's aspect ratio.
IMG_W, IMG_H = 700, 900


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()

    entries: list[dict] = []
    page = 1
    while len(entries) < args.limit * 2:  # over-fetch, then sample
        print(f"Fetching list page {page} ...")
        try:
            resp = session.get(LIST_URL, params={"page": page, "limit": 100}, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  ! list page {page} failed: {exc}", file=sys.stderr)
            break
        if not isinstance(batch, list) or not batch:
            break
        entries.extend(batch)
        page += 1
        sleep_polite(0.2)

    print(f"Got {len(entries)} photo entries.")
    if not entries:
        sys.exit("No photos returned — check https://picsum.photos/ is reachable.")

    random.seed(args.seed)
    random.shuffle(entries)

    saved = 0
    for e in entries:
        if saved >= args.limit:
            break
        pid = str(e.get("id"))
        if not pid:
            continue
        url = f"https://picsum.photos/id/{pid}/{IMG_W}/{IMG_H}"
        dest = OUT_DIR / f"picsum_{pid}.jpg"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="",
                source=f"picsum:{pid}",
                license_note="Lorem Picsum / Unsplash — Unsplash License, free to use",
                notes=str(e.get("author", "")),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
